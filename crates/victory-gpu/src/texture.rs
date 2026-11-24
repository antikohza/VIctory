//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use victory_formats::texture::{Error, Layout};
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

pub(crate) fn auto_mip_drop_for_memory(vram_bytes: u64) -> u32 {
    const EIGHT_GB_THRESHOLD: u64 = 8 * 1024 * 1024 * 1024 + 512 * 1024 * 1024;
    u32::from(vram_bytes != 0 && vram_bytes <= EIGHT_GB_THRESHOLD)
}

/// The number of most detailed mips to drop for sampled host textures.
/// 0: keep original resolution.
/// 1: drop 1 mip (halves dimensions, ~75% VRAM savings).
/// 2: drop 2 mips (quarters dimensions, ~93.75% VRAM savings).
/// 3: drop 3 mips (eighths dimensions, ~98.4% VRAM savings).
///
/// When `VICTORY_TEXTURE_MIP_DROP` is not explicitly set in the environment,
/// this automatically defaults to 1 on adapters with <= 8 GB dedicated VRAM
/// to prevent host driver committed out-of-memory crashes on 6 GB and 8 GB GPUs.
pub(crate) fn texture_mip_drop() -> u32 {
    if let Some(v) = std::env::var("VICTORY_TEXTURE_MIP_DROP")
        .ok()
        .and_then(|val| val.parse::<u32>().ok())
    {
        return v.min(3);
    }
    let own = crate::host::device_memory();
    let drop = auto_mip_drop_for_memory(own);
    if drop > 0 {
        once!(
            "auto-enabling VICTORY_TEXTURE_MIP_DROP=1 for <= 8 GB GPU (vram_mb={})",
            own / (1024 * 1024)
        );
    }
    drop
}

/// The minimum texture dimension (both width and height) eligible for mip-dropping.
/// Defaults to 1024. Textures smaller than this on either axis (or under 2048 if configured)
/// are preserved at full resolution. Configurable via `VICTORY_TEXTURE_MIP_DROP_MIN_SIZE`.
pub(crate) fn texture_mip_drop_min_size() -> u64 {
    static MIN_SIZE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MIN_SIZE.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_MIP_DROP_MIN_SIZE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1024)
    })
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    size: usize,
    data: *mut u8,
    footprints: Vec<UploadFootprint>,
    layout: Layout,
    /// Mips per array slice.
    mips: u32,
    /// Dropped mips count (offset into console mips for detiling).
    pub(crate) mip_drop: u32,
    memory: *const u8,
    last: Mutex<Last>,
}

struct UploadFootprint {
    placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    range: Range<usize>,
    depth_pitch: usize,
}

impl UploadFootprint {
    fn new(
        placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        rows: u32,
        row_size: u64,
        total: usize,
    ) -> Option<Self> {
        let pitch = placed.Footprint.RowPitch as usize;
        let row_size = usize::try_from(row_size).ok()?;
        let (len, depth_pitch) = footprint_size(pitch, rows, placed.Footprint.Depth, row_size)?;
        let start = usize::try_from(placed.Offset).ok()?;
        let end = start.checked_add(len)?;
        (end <= total).then_some(Self {
            placed,
            range: start..end,
            depth_pitch,
        })
    }
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after,
/// with room to read the memory into.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    read: Vec<u8>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

thread_local! {
    /// Texture memory read for hashing and detiling: one per submitting
    /// thread, as large as the largest texture it read.
    static UNWATCHED_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Subresource-level tracking for host GPU writes (copies / resolves) so
/// partial copies across mismatched mip-drops do not suppress title uploads
/// of untouched destination subresources. Uses a zero-allocation `u128` bitmask
/// for typical textures (<128 subresources).
#[derive(Clone)]
enum HostWritten {
    All,
    Mask(u128),
    Many(std::collections::HashSet<u32>),
}

impl HostWritten {
    fn contains(&self, subresource: u32) -> bool {
        match self {
            Self::All => true,
            Self::Mask(mask) => subresource < u128::BITS && (*mask & (1u128 << subresource)) != 0,
            Self::Many(set) => set.contains(&subresource),
        }
    }

    fn all_written(&self, count: u32) -> bool {
        match self {
            Self::All => true,
            Self::Mask(mask) if count <= u128::BITS => {
                let needed = if count == u128::BITS {
                    u128::MAX
                } else {
                    (1u128 << count) - 1
                };
                (*mask & needed) == needed
            }
            Self::Mask(_) => false,
            Self::Many(set) => (0..count).all(|s| set.contains(&s)),
        }
    }
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<HashMap<usize, HostWritten>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    map.get_or_insert_with(Default::default)
        .insert(resource as usize, HostWritten::All);
}

/// A partial host copy or resolve writes a single host subresource of
/// `resource`, leaving untouched subresources eligible for title upload.
pub(crate) fn host_written_subresource(resource: This, host_subresource: u32) {
    let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    let entry = map
        .get_or_insert_with(Default::default)
        .entry(resource as usize)
        .or_insert(HostWritten::Mask(0));
    match entry {
        HostWritten::All => {}
        HostWritten::Mask(mask) if host_subresource < u128::BITS => {
            *mask |= 1u128 << host_subresource;
        }
        HostWritten::Mask(mask) => {
            let mut set: std::collections::HashSet<u32> = (0..u128::BITS)
                .filter(|b| (*mask & (1u128 << b)) != 0)
                .collect();
            set.insert(host_subresource);
            *entry = HostWritten::Many(set);
        }
        HostWritten::Many(set) => {
            set.insert(host_subresource);
        }
    }
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut written = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(written) = written.as_mut() {
        written.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    // SAFETY: a live resource.
    title_written_flags(unsafe { texture.GetDesc() }.Flags)
}

fn title_written_flags(flags: D3D12_RESOURCE_FLAGS) -> bool {
    let host_writable = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
        | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    (flags & host_writable) == D3D12_RESOURCE_FLAG_NONE
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let written = {
        let map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        match map.as_ref().and_then(|m| m.get(&(resource as usize))) {
            Some(HostWritten::All) => return,
            Some(HostWritten::Mask(m)) => Some(HostWritten::Mask(*m)),
            Some(HostWritten::Many(s)) => Some(HostWritten::Many(s.clone())),
            None => None,
        }
    };
    // shortcut: subresources the host copied into keep the host's contents
    // even if the title later writes their memory too.
    if !list.first_use(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    if written
        .as_ref()
        .is_some_and(|w| w.all_written(staging.footprints.len() as u32))
    {
        return;
    }
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        if written.as_ref().is_some_and(|w| w.contains(i as u32)) {
            continue;
        }
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.placed.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.placed.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.placed.Offset, footprint.range.len() as u64),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint.placed,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let Ok(size) = usize::try_from(self.layout.size) else {
            return 0;
        };
        let start = self.memory as usize;
        let Some(end) = start
            .checked_add(size)
            .filter(|_| size <= isize::MAX as usize)
        else {
            return 0;
        };
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, end, false) {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            self.detile(&last.read);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return 0;
        }
        // Read through the kernel, as above: the title may unmap the pages
        // (evicting a texture a big table lists) while we read them, and a
        // direct read would fault. The copy is reused, not kept per texture.
        UNWATCHED_SCRATCH.with_borrow_mut(|tiled| {
            tiled.resize(size, 0);
            crate::host::copy_mapped(tiled.as_mut_ptr(), self.memory, size, false);
            self.changed(last, tiled)
        })
    }

    /// After reading the texture's memory into `tiled`: detiles it unless
    /// its hash is the last one's. Returns the bytes read.
    fn changed(&self, last: &mut Last, tiled: &[u8]) -> usize {
        let size = tiled.len();
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        // SAFETY: the upload buffer is mapped for exactly this range and
        // refresh holds `last`'s lock while writing it.
        let linear = unsafe { std::slice::from_raw_parts_mut(self.data, self.size) };
        if let Err((e, mip, slice)) = detile_plane_with_drop(
            &self.layout,
            self.mips,
            self.mip_drop,
            &self.footprints,
            tiled,
            linear,
        ) {
            tracing::warn!(?e, mip, slice, "texture not detiled");
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn detile_plane(
    layout: &Layout,
    mips: u32,
    footprints: &[UploadFootprint],
    tiled: &[u8],
    linear: &mut [u8],
) -> Result<(), (Error, u32, u32)> {
    detile_plane_with_drop(layout, mips, 0, footprints, tiled, linear)
}

fn detile_plane_with_drop(
    layout: &Layout,
    mips: u32,
    mip_drop: u32,
    footprints: &[UploadFootprint],
    tiled: &[u8],
    linear: &mut [u8],
) -> Result<(), (Error, u32, u32)> {
    for (i, footprint) in footprints.iter().enumerate() {
        let (host_mip, slice) = (i as u32 % mips, i as u32 / mips);
        let mip = host_mip + mip_drop;
        let target =
            linear
                .get_mut(footprint.range.clone())
                .ok_or((Error::TooSmall, mip, slice))?;
        layout
            .detile(
                tiled,
                mip,
                slice,
                target,
                footprint.placed.Footprint.RowPitch as usize,
                footprint.depth_pitch,
            )
            .map_err(|e| (e, mip, slice))?;
    }
    Ok(())
}

/// Bytes read by a copy and the stride between depth slices.
fn footprint_size(pitch: usize, rows: u32, depth: u32, row_size: usize) -> Option<(usize, usize)> {
    if row_size == 0 || row_size > pitch {
        return None;
    }
    let depth_pitch = pitch.checked_mul(rows as usize)?;
    // D3D12 allocates the data bytes of the last row only.
    let preceding_rows = (rows as usize)
        .checked_mul(depth as usize)?
        .checked_sub(1)?;
    let len = pitch.checked_mul(preceding_rows)?.checked_add(row_size)?;
    Some((len, depth_pitch))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        let Some(next) = (info.BaseAddress as usize).checked_add(info.RegionSize) else {
            return false;
        };
        p = next;
    }
    true
}

fn plane_subresources(desc: &D3D12_RESOURCE_DESC) -> u32 {
    let mips = u32::from(desc.MipLevels);
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        mips
    } else {
        mips * u32::from(desc.DepthOrArraySize)
    }
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let mip_drop = u32::from(desc.mip_levels).saturating_sub(u32::from(d.MipLevels));
    let count = plane_subresources(&d);
    if count == 0 {
        return None;
    }
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut row_sizes = vec![0u64; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&mut total),
        );
    }
    let size = usize::try_from(total)
        .ok()
        .filter(|&size| size <= isize::MAX as usize)?;
    let footprints = footprints
        .into_iter()
        .zip(rows)
        .zip(row_sizes)
        .map(|((placed, rows), row_size)| UploadFootprint::new(placed, rows, row_size, size))
        .collect::<Option<Vec<_>>>()?;
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, mip_drop, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        size,
        data,
        footprints,
        layout,
        mips: u32::from(d.MipLevels),
        mip_drop,
        memory,
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_R8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM,
    };

    #[test]
    fn footprint_ends_after_the_last_rows_data() {
        // Two depth slices of three rows. The sixth row starts at 1280,
        // but only its 20 data bytes belong to the upload range.
        assert_eq!(footprint_size(256, 3, 2, 20), Some((1300, 768)));
    }

    #[test]
    fn footprint_rejects_invalid_geometry_and_out_of_bounds_placement() {
        assert_eq!(footprint_size(256, 0, 1, 20), None);
        assert_eq!(footprint_size(256, 1, 0, 20), None);
        assert_eq!(footprint_size(256, 1, 1, 0), None);
        assert_eq!(footprint_size(256, 1, 1, 257), None);
        assert_eq!(footprint_size(usize::MAX, 1, 2, 1), None);

        let placed = placed(512, [5, 3, 2]);
        let footprint = UploadFootprint::new(placed, 3, 20, 1812).unwrap();
        assert_eq!(footprint.range, 512..1812);
        assert!(UploadFootprint::new(placed, 3, 20, 1811).is_none());
        let overflowing = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: u64::MAX,
            ..placed
        };
        assert!(UploadFootprint::new(overflowing, 3, 20, usize::MAX).is_none());
    }

    #[test]
    fn plane_zero_counts_each_array_mip_and_each_volume_mip_once() {
        let mut desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            DepthOrArraySize: 6,
            MipLevels: 4,
            ..Default::default()
        };
        assert_eq!(plane_subresources(&desc), 24);
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        assert_eq!(plane_subresources(&desc), 24);
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        assert_eq!(plane_subresources(&desc), 4);
    }

    fn placed(offset: u64, [width, height, depth]: [u32; 3]) -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: offset,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Width: width,
                Height: height,
                Depth: depth,
                RowPitch: 256,
                ..Default::default()
            },
        }
    }

    // Guest mip and slice, host offset and dimensions, then rows and bytes per row.
    type Shape = ((u32, u32), u64, [u32; 3], u32, u64);

    fn assert_detiles(desc: TextureDesc, element_bytes: usize, shapes: &[Shape], total: usize) {
        let layout = &Texture::new(&desc).unwrap().planes[0].layout;
        let mut tiled = vec![0u8; layout.size as usize];
        let mut linear = vec![0xff; total];
        let mut expected = linear.clone();
        let mut footprints = Vec::new();
        for (i, &((mip, slice), offset, dims, rows, row_size)) in shapes.iter().enumerate() {
            footprints
                .push(UploadFootprint::new(placed(offset, dims), rows, row_size, total).unwrap());
            for z in 0..dims[2] {
                for y in 0..rows {
                    for x in 0..row_size as usize / element_bytes {
                        let guest = layout.offset(mip, x as u32, y, slice + z).unwrap() as usize;
                        let host =
                            offset as usize + (z * rows + y) as usize * 256 + x * element_bytes;
                        let value = (i * 37 + (z * 7 + y * 13) as usize + x * 3) as u8;
                        for b in 0..element_bytes {
                            tiled[guest + b] = value + b as u8;
                            expected[host + b] = value + b as u8;
                        }
                    }
                }
            }
        }
        detile_plane(layout, desc.mips, &footprints, &tiled, &mut linear).unwrap();
        assert_eq!(linear, expected);
    }

    #[test]
    fn array_upload_uses_mip_first_order_and_preserves_padding() {
        let desc = TextureDesc {
            dimension: Dimension::Tex2d,
            width: 7,
            height: 5,
            depth_or_array: 2,
            mips: 2,
            format: DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            samples: 1,
            swizzle: Swizzle::SW_4KB_S,
        };
        assert_detiles(
            desc,
            4,
            &[
                ((0, 0), 0, [7, 5, 1], 5, 28),
                ((1, 0), 1536, [3, 2, 1], 2, 12),
                ((0, 1), 2048, [7, 5, 1], 5, 28),
                ((1, 1), 3584, [3, 2, 1], 2, 12),
            ],
            3852,
        );
    }

    #[test]
    fn volume_upload_uses_pitched_depth_slices_and_shrinks_depth_by_mip() {
        let desc = TextureDesc {
            dimension: Dimension::Tex3d,
            width: 5,
            height: 3,
            depth_or_array: 3,
            mips: 2,
            format: DXGI_FORMAT_R8_UNORM.0 as u32,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        };
        assert_detiles(
            desc,
            1,
            &[
                ((0, 0), 0, [5, 3, 3], 3, 5),
                ((1, 0), 2560, [2, 1, 1], 1, 2),
            ],
            2562,
        );
    }

    #[test]
    fn block_compressed_upload_uses_whole_blocks_for_partial_rows_and_mips() {
        for (format, block_bytes) in [(DXGI_FORMAT_BC1_UNORM, 8), (DXGI_FORMAT_BC7_UNORM, 16)] {
            let desc = TextureDesc {
                dimension: Dimension::Tex2d,
                width: 7,
                height: 5,
                depth_or_array: 1,
                mips: 2,
                format: format.0 as u32,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            };
            assert_detiles(
                desc,
                block_bytes as usize,
                &[
                    ((0, 0), 0, [8, 8, 1], 2, block_bytes * 2),
                    ((1, 0), 512, [4, 4, 1], 1, block_bytes),
                ],
                512 + block_bytes as usize,
            );
        }
    }

    #[test]
    fn host_writable_flags_exclude_texture_uploads() {
        assert!(title_written_flags(D3D12_RESOURCE_FLAG_NONE));
        assert!(title_written_flags(
            D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE
        ));
        for flag in [
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ] {
            assert!(!title_written_flags(flag));
            assert!(!title_written_flags(
                flag | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE
            ));
        }
    }

    #[test]
    fn committed_rejects_an_address_range_that_wraps() {
        assert!(!committed(std::ptr::without_provenance(usize::MAX), 2));
        assert!(committed(std::ptr::without_provenance(usize::MAX), 0));
        assert!(!committed(std::ptr::null(), 1));
    }

    #[test]
    fn committed_checks_every_region_in_the_range() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };

        // SAFETY: reserve two pages, then commit them separately with
        // different protections so VirtualQuery reports distinct regions.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 8192, MEM_RESERVE, PAGE_READWRITE);
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base, 4096, MEM_COMMIT, PAGE_READWRITE), base);
            let bytes = base.cast::<u8>();
            assert!(committed(bytes, 4096));
            assert!(!committed(bytes.add(4095), 2));
            let second = bytes.add(4096).cast();
            assert_eq!(
                VirtualAlloc(second, 4096, MEM_COMMIT, PAGE_READONLY),
                second
            );
            assert!(committed(bytes.add(4095), 2));
            assert!(committed(bytes, 8192));
            assert_ne!(VirtualFree(base, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn texture_mip_drop_runs_without_panic() {
        let _ = texture_mip_drop();
    }

    #[test]
    fn auto_mip_drop_threshold_selects_low_vram_gpus() {
        assert_eq!(auto_mip_drop_for_memory(0), 0);
        assert_eq!(auto_mip_drop_for_memory(128 << 20), 1); // 128 MiB iGPU
        assert_eq!(auto_mip_drop_for_memory(6 << 30), 1); // 6 GB RTX 3050
        assert_eq!(auto_mip_drop_for_memory(8_273_264_640), 1); // 8 GB RTX 5070 Laptop
        assert_eq!(auto_mip_drop_for_memory(12 << 30), 0); // 12 GB RTX 4070 SUPER
        assert_eq!(auto_mip_drop_for_memory(16 << 30), 0); // 16 GB RX 9060 XT
    }

    #[test]
    fn host_written_tracks_subresources_and_many_promotion() {
        let key = 0xDEAD_BEEFusize as This;
        forget(key as usize);

        host_written_subresource(key, 2);
        host_written_subresource(key, 5);
        {
            let map = HOST_WRITTEN.lock().unwrap();
            let entry = map.as_ref().unwrap().get(&(key as usize)).unwrap();
            assert!(!entry.contains(0));
            assert!(entry.contains(2));
            assert!(entry.contains(5));
            assert!(!entry.all_written(6));
        }

        // Promotion past u128::BITS keeps earlier subresource bits intact.
        host_written_subresource(key, 200);
        {
            let map = HOST_WRITTEN.lock().unwrap();
            let entry = map.as_ref().unwrap().get(&(key as usize)).unwrap();
            assert!(entry.contains(2));
            assert!(entry.contains(5));
            assert!(entry.contains(200));
            assert!(!entry.contains(1));
        }

        host_written(key);
        {
            let map = HOST_WRITTEN.lock().unwrap();
            let entry = map.as_ref().unwrap().get(&(key as usize)).unwrap();
            assert!(entry.contains(0));
            assert!(entry.all_written(256));
        }
        forget(key as usize);
    }
}

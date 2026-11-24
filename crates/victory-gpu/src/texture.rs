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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    /// Dropped mips count (offset into console mips for detiling).
    pub(crate) mip_drop: u32,
    memory: *const u8,
    last: Mutex<Last>,
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
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
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
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
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
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
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
                PlacedFootprint: *footprint,
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
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, start + size, false) {
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
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
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
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (host_mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let console_mip = host_mip + self.mip_drop;
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, console_mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, console_mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
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
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
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
            None,
            Some(&mut total),
        );
    }
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
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
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


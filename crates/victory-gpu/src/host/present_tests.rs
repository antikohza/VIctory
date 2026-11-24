//! Run by itself on an interactive Windows desktop:
//! `cargo test -p victory-gpu --features diagnostics synthetic_presentx -- --ignored`.
//! Set `VICTORY_TEST_GPU_VENDOR` to a hex PCI vendor ID for explicit hardware
//! selection, or `VICTORY_D3D12_WARP=1` for the software adapter.

use super::*;

const ORIGINAL_SIZE: (u32, u32) = (384, 216);
const RESIZED_SIZE: (u32, u32) = (320, 180);

#[test]
#[ignore = "requires an interactive Windows desktop and D3D12"]
fn synthetic_presentx() {
    use crate::object::{HRESULT, Iid, S_OK, This, slot};
    use crate::resource::{HeapProperties, PAGEABLE_IIDS, ResourceDesc};
    use std::ptr::null;

    ACTIVE.set(true);
    let host = host().expect("D3D12 device");
    let mut device = null();
    let iid = iid_bytes(ID3D12Device11::IID);
    assert_eq!(
        crate::device::create(null(), &crate::device::CHECKED_VERSION, &iid, &mut device),
        S_OK
    );
    let iid = &PAGEABLE_IIDS[0];

    let mut allocator = null();
    let mut queue = null();
    let mut list = null();
    let make_allocator =
        slot::<extern "system" fn(This, u32, *const Iid, *mut This) -> HRESULT>(device, 9 * 8);
    let make_queue = slot::<
        extern "system" fn(This, *const [u32; 4], *const Iid, *mut This) -> HRESULT,
    >(device, 48 * 8);
    let make_list = slot::<
        extern "system" fn(This, *const [u32; 4], This, This, *const Iid, *mut This) -> HRESULT,
    >(device, 73 * 8);
    assert_eq!(make_allocator(device, 0, iid, &mut allocator), S_OK);
    assert_eq!(make_queue(device, &[0; 4], iid, &mut queue), S_OK);
    let list_iid = iid_bytes(ID3D12GraphicsCommandList::IID);
    assert_eq!(
        make_list(device, &[0; 4], allocator, null(), &list_iid, &mut list),
        S_OK
    );
    let clear = slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const std::ffi::c_void)>(
        list,
        52 * 8,
    );
    let close = slot::<extern "system" fn(This) -> HRESULT>(list, 9 * 8);
    let reset = slot::<extern "system" fn(This, This, This, u32) -> HRESULT>(list, 139 * 8);
    let submit = slot::<extern "system" fn(This, u32, *const This, u64, u32)>(queue, 35 * 8);

    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    // SAFETY: the console heap descriptor has the PC layout.
    let heap: HeapProperties = unsafe { std::mem::transmute(heap) };
    let make_target = |(width, height)| {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        // SAFETY: the console resource descriptor has the PC layout.
        let desc: ResourceDesc = unsafe { std::mem::transmute(desc) };
        let mut target = null();
        assert_eq!(
            crate::resource::create_committed(device, &heap, 0, &desc, 0, null(), iid, &mut target),
            S_OK
        );
        assert!(crate::resource::host_resource(target).is_some());
        target
    };
    let original = make_target(ORIGINAL_SIZE);
    let resized = make_target(RESIZED_SIZE);

    let vs = shader(true);
    let ps = shader(false);
    // Stream subobjects have eight-byte alignment, including the shader pointers.
    let mut stream: Vec<u64> = vec![
        1,
        vs.as_ptr() as u64,
        vs.len() as u64,
        2,
        ps.as_ptr() as u64,
        ps.len() as u64,
        14 | (3 << 32),
    ];
    let mut state = [0u32; 10];
    state[0] = 15; // D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS
    state[1] = DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32;
    state[9] = 1;
    stream.extend(
        state
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u64::from(p[0]) | (u64::from(p[1]) << 32)),
    );
    let depth_disabled = [11u64, 0, 0, 0, 0, 0, 0];
    stream.extend(depth_disabled);
    let desc = [stream.len() * 8, stream.as_ptr() as usize];
    let mut pipeline = null();
    let create_stream = slot::<
        extern "system" fn(This, *const [usize; 2], *const Iid, *mut This) -> HRESULT,
    >(device, 86 * 8);
    assert_eq!(create_stream(device, &desc, iid, &mut pipeline), S_OK);
    assert!(!pipeline.is_null());
    let mut view = [0u64; 14];
    let handle = view.as_mut_ptr() as u64;
    for (frame, (target, (width, height))) in [
        (original, ORIGINAL_SIZE),
        (original, ORIGINAL_SIZE),
        (resized, RESIZED_SIZE),
        (original, ORIGINAL_SIZE),
    ]
    .into_iter()
    .enumerate()
    {
        if frame != 0 {
            assert_eq!(reset(list, allocator, null(), 0), S_OK);
        }
        crate::descriptor::create_rtv(device, target, null(), handle);
        // A distinct color proves that returning to the original extent
        // presents fresh content from the reused target.
        let color = if frame == 3 {
            [0.0, 0.0, 1.0, 1.0]
        } else {
            [1.0, 0.0, 1.0, 1.0]
        };
        clear(list, handle, &color, 0, null());
        if matches!(frame, 1 | 2) {
            let viewport = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let scissor = RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            slot::<extern "system" fn(This, u32)>(list, 20 * 8)(list, 4);
            slot::<extern "system" fn(This, u32, *const D3D12_VIEWPORT)>(list, 21 * 8)(
                list, 1, &viewport,
            );
            slot::<extern "system" fn(This, u32, *const RECT)>(list, 22 * 8)(list, 1, &scissor);
            slot::<extern "system" fn(This, This)>(list, 25 * 8)(list, pipeline);
            slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 50 * 8)(
                list,
                1,
                &handle,
                0,
                null(),
            );
            let before = crate::draw::drawn();
            slot::<extern "system" fn(This, u32, u32, u32, u32)>(list, 12 * 8)(list, 3, 1, 0, 0);
            assert_eq!(crate::draw::drawn(), before + 1);
        }
        assert_eq!(close(list), S_OK);
        submit(queue, 1, &list, frame as u64, 0);
        // SAFETY: the host device remains live until process exit.
        unsafe { host.device.GetDeviceRemovedReason() }.expect("device still alive");
        {
            #[repr(C)]
            struct Plane {
                token: u64,
                count: u32,
                resources: *const This,
            }
            let plane = Plane {
                token: frame as u64,
                count: 1,
                resources: &target,
            };
            let present = slot::<extern "system" fn(This, u32, *const Plane, *const u8) -> HRESULT>(
                queue,
                32 * 8,
            );
            assert_eq!(present(queue, 1, &plane, null()), S_OK);
            assert_eq!(crate::queue::presented(), frame as u64 + 1);
            let guard = host.presenter.lock().unwrap();
            let presenter = guard.as_ref().expect("swap chain created");
            // SAFETY: the presenter owns the live swap chain.
            let next = unsafe { presenter.swap_chain.GetCurrentBackBufferIndex() };
            let previous = if frame >= 2 { 0 } else { frame as u32 };
            assert_eq!(next, (previous + 1) % BUFFERS);
        }
    }
    ACTIVE.set(false);
    for object in [list, queue, allocator, original, resized, pipeline] {
        slot::<extern "system" fn(This) -> u32>(object, 16)(object);
    }
}
thread_local! {
    pub(super) static ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn iid_bytes(guid: windows::core::GUID) -> crate::object::Iid {
    // SAFETY: GUID is a 16-byte repr(C) struct without padding.
    unsafe { std::mem::transmute(guid) }
}

pub(super) fn adapter() -> windows::core::Result<Option<windows::core::IUnknown>> {
    let Some(vendor) = std::env::var("VICTORY_TEST_GPU_VENDOR")
        .ok()
        .filter(|_| ACTIVE.get())
    else {
        return Ok(None);
    };
    let vendor = u32::from_str_radix(&vendor, 16).expect("expected a hex PCI vendor ID");
    // SAFETY: DXGI creates and returns owned COM interfaces.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        for index in 0.. {
            let adapter = factory.EnumAdapters1(index)?;
            let desc = adapter.GetDesc1()?;
            if desc.VendorId == vendor {
                eprintln!(
                    "adapter: {}",
                    String::from_utf16_lossy(&desc.Description).trim_end_matches('\0')
                );
                return adapter.cast().map(Some);
            }
        }
    }
    unreachable!()
}

pub(super) fn check_pixels(
    frame: u64,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let expected = match frame {
        1 => [255, 0, 255, 255],
        2 | 3 => [0, 255, 0, 255],
        4 => [0, 0, 255, 255],
        _ => panic!("unexpected frame {frame}"),
    };
    let layout = footprint.Footprint;
    let size = if frame == 3 {
        RESIZED_SIZE
    } else {
        ORIGINAL_SIZE
    };
    assert_eq!((layout.Width, layout.Height), size);
    let mut data = std::ptr::null_mut();
    // SAFETY: the readback copy has completed; the footprint describes the allocation.
    unsafe {
        buffer
            .Map(0, None, Some(&mut data))
            .expect("map back buffer");
        assert!(!data.is_null());
        for y in 0..layout.Height as usize {
            let offset = footprint.Offset as usize + y * layout.RowPitch as usize;
            let row = std::slice::from_raw_parts(
                data.cast::<u8>().add(offset),
                layout.Width as usize * 4,
            );
            assert!(
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .all(|pixel| *pixel == expected),
                "swap-chain frame {frame}, row {y}"
            );
        }
        buffer.Unmap(0, None);
    }
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn signature(name: &str, kind: u32, component: u32, mask: u32) -> Vec<u8> {
    let mut signature = words(&[1, 8, 0, 40, 0, kind, component, 0, mask, 0]);
    signature.extend(name.bytes());
    signature.push(0);
    signature
}

fn shader(vertex: bool) -> Vec<u8> {
    let (slot, code, input, output) = if vertex {
        (
            0x28,
            words(&[
                0x7e0a02ff, 0x40400000, // v5 = 3.0
                0x7d840081, 0x02020af3, // x = id == 1 ? 3 : -1
                0x7e0a02ff, 0xc0400000, // v5 = -3.0
                0x7d840082, 0x02040af2, // y = id == 2 ? -3 : 1
                0x7e060280, 0x7e0802f2, // z = 0, w = 1
                0xf80008cf, 0x04030201, 0xbf810000, // export position and end
            ]),
            signature("SV_VertexID", 6, 1, 0x101),
            signature("SV_Position", 1, 3, 15),
        )
    } else {
        (
            0x30,
            words(&[
                0x7e000280, 0x7e0202f2, // v0 = 0, v1 = 1
                0xf800180f, 0x01000100, 0xbf810000, // export green and end
            ]),
            words(&[0, 8]),
            signature("SV_Target", 64, 3, 15),
        )
    };

    // One FSSM stage record with no resources, semantics or export patches.
    const RECORD: usize = 0x68;
    const STAGE: usize = RECORD + 0x54;
    const PATCHES: usize = 0x14c;
    const PATCH_SIZE: usize = 404;
    const CODE: usize = STAGE + PATCHES + 8 * PATCH_SIZE;
    let mut payload = vec![0; CODE];
    payload[..8].copy_from_slice(&6u64.to_le_bytes());
    for offset in [8, 16, 24, 32, 40, 48, 72, 96] {
        payload[offset..offset + 8].copy_from_slice(&0xdead_beef_dead_beefu64.to_le_bytes());
    }
    payload[slot..slot + 8].copy_from_slice(&(RECORD as u64).to_le_bytes());
    payload[RECORD..RECORD + 8].copy_from_slice(&(CODE as u64).to_le_bytes());
    payload[RECORD + 8..RECORD + 16].copy_from_slice(&(code.len() as u64).to_le_bytes());
    payload[RECORD + 0x14] = 1 << 5; // wave32
    payload[RECORD + 0x50] = 8; // VGPR count
    for target in 0..8 {
        let offset = STAGE + PATCHES + target * PATCH_SIZE;
        payload[offset..offset + 4].fill(0xff);
    }
    payload.extend(code);
    let mut fssm = b"FSSM".to_vec();
    fssm.extend(0x101u32.to_le_bytes());
    fssm.extend((payload.len() as u64).to_le_bytes());
    fssm.extend(payload);

    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    xbch.extend(3u32.to_le_bytes());
    xbch.extend((compressed.len() as u64).to_le_bytes());
    xbch.extend((fssm.len() as u64).to_le_bytes());
    xbch.extend(compressed);
    let mut dxil = vec![0; 24];
    dxil.extend(b"XDAT");
    dxil.extend(words(&[1, 0x24, 0x0d, 48, xbch.len() as u32]));
    dxil.extend(xbch);

    let mut container = b"DXBC".to_vec();
    container.resize(44, 0);
    container[20] = 1; // container version
    container[28] = 3; // part count
    for (index, (tag, data)) in [(b"ISG1", input), (b"OSG1", output), (b"DXIL", dxil)]
        .into_iter()
        .enumerate()
    {
        let offset = (container.len() as u32).to_le_bytes();
        container[32 + 4 * index..36 + 4 * index].copy_from_slice(&offset);
        container.extend(tag);
        container.extend((data.len() as u32).to_le_bytes());
        container.extend(data);
        container.resize(container.len().next_multiple_of(4), 0);
    }
    let size = (container.len() as u32).to_le_bytes();
    container[24..28].copy_from_slice(&size);
    container
}

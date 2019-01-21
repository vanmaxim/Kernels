//
//  select_index.metal
//  select_index
//
//  Created by Maksim on 1/17/19.
//  Copyright © 2019 Mapbox. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void select_index(texture2d_array<half, access::read> inTexture          [[texture(0)]],
                         texture2d_array<half, access::read_write> outTexture   [[texture(1)]],
                         constant int4 *indecies                                [[buffer(0)]],
                         ushort3 gid                                            [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height())
        return;
    
    int const z = static_cast<int>(gid.z);
    int4 const i = indecies[z];
    
    int const z0 = i[0] / 4;
    int const j0 = i[0] % 4;
    
    int const z1 = i[1] / 4;
    int const j1 = i[1] % 4;
    
    int const z2 = i[2] / 4;
    int const j2 = i[2] % 4;
    
    int const z3 = i[3] / 4;
    int const j3 = i[3] % 4;
    
    half const o0 = inTexture.read(gid.xy, z0)[j0];
    half const o1 = inTexture.read(gid.xy, z1)[j1];
    half const o2 = inTexture.read(gid.xy, z2)[j2];
    half const o3 = inTexture.read(gid.xy, z3)[j3];
    
    half4 const out = half4(o0, o1, o2, o3);
    outTexture.write(out, gid.xy, gid.z);
}

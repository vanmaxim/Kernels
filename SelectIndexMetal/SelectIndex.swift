//
//  SelectIndex.swift
//  select_index
//
//  Created by Maksim on 1/21/19.
//  Copyright © 2019 Mapbox. All rights reserved.
//

import Foundation
import simd
import Metal
import CoreML

@objc(SelectIndex) class SelectIndex: NSObject, MLCustomLayer {
    
    let device = MTLCreateSystemDefaultDevice()!
    let library = MTLCreateSystemDefaultDevice()!.makeDefaultLibrary()!
    let commandQueue = MTLCreateSystemDefaultDevice()!.makeCommandQueue()!
    
    let axis: Int
    var indecies: [int4]!
    let gatherPipeline: MTLComputePipelineState
    
    required init(parameters: [String : Any]) throws {
        axis = parameters["axis"] as! Int
        let function = library.makeFunction(name: "select_index")!
        gatherPipeline = try! device.makeComputePipelineState(function: function)
    }
    
    func setWeightData(_ weights: [Data]) throws {
        indecies = weights[0].withUnsafeBytes {
            Array(UnsafeBufferPointer<Float>(start: $0, count: weights[0].count / MemoryLayout<Float>.stride))
            }.chunked(into: 4).map { int4(Int32($0[0]), Int32($0[1]), Int32($0[2]), Int32($0[3])) }
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        return inputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        
    }
    
    func encode(commandBuffer: MTLCommandBuffer, inputs: [MTLTexture], outputs: [MTLTexture]) throws {
        let buffer = device.makeBuffer(bytes: indecies, length: indecies.count * MemoryLayout<int4>.stride, options: .storageModeShared)!
        
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.label = "Gather"
        for i in 0..<inputs.count {
            encoder.setTexture(inputs[i], index: 0)
            encoder.setTexture(outputs[i], index: 1)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.dispatch(pipeline: gatherPipeline, texture: inputs[i])
            encoder.endEncoding()
        }
    }
}

extension MTLComputeCommandEncoder {
    public func dispatch(pipeline: MTLComputePipelineState, texture: MTLTexture) {
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let threadGroupSize = MTLSizeMake(w, h, 1)
        
        let threadGroups = MTLSizeMake(
            (texture.width       + threadGroupSize.width  - 1) / threadGroupSize.width,
            (texture.height      + threadGroupSize.height - 1) / threadGroupSize.height,
            (texture.arrayLength + threadGroupSize.depth  - 1) / threadGroupSize.depth)
        
        setComputePipelineState(pipeline)
        dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }
}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

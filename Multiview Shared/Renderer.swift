//
//  Renderer.swift
//  Multiview Shared
//
//  Created by we on 2022/04/08.
//

// Our platform independent renderer class

import Metal
import MetalKit
import simd

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100

let maxBuffersInFlight = 3

enum RendererError: Error {
    case badVertexDescriptor
}

class Renderer: NSObject, MTKViewDelegate {
    
    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicUniformBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var leftPipelineState: MTLRenderPipelineState
    var rearPipelineState: MTLRenderPipelineState
    var bottomPipelineState: MTLRenderPipelineState
    var depthState: MTLDepthStencilState
    var colorMap: MTLTexture
    
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    
    var uniformBufferOffset = 0
    
    var uniformBufferIndex = 0
    
    var uniforms0: UnsafeMutablePointer<Uniforms>
    var uniforms1: UnsafeMutablePointer<Uniforms>
    var uniforms2: UnsafeMutablePointer<Uniforms>
    var uniforms3: UnsafeMutablePointer<Uniforms>
    
    var projectionMatrix: matrix_float4x4 = matrix_float4x4()
    
    var rotation: Float = 0
    
    var mesh: MTKMesh
    
    init?(metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        
        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight * 4
        
        guard let buffer = self.device.makeBuffer(length:uniformBufferSize, options:[MTLResourceOptions.storageModeShared]) else { return nil }
        dynamicUniformBuffer = buffer
        
        self.dynamicUniformBuffer.label = "UniformBuffer"
        
        let uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to:Uniforms.self, capacity:4)
        uniforms0 = uniforms.advanced(by: 0)
        uniforms1 = uniforms.advanced(by: 1)
        uniforms2 = uniforms.advanced(by: 2)
        uniforms3 = uniforms.advanced(by: 3)
        
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.sampleCount = 1
        
        let mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()
        
        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor,
                                                                       vertexFunction: "vertexShader")
            leftPipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                           metalKitView: metalKitView,
                                                                           mtlVertexDescriptor: mtlVertexDescriptor,
                                                                           vertexFunction: "vertexLeftShader")
            rearPipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                           metalKitView: metalKitView,
                                                                           mtlVertexDescriptor: mtlVertexDescriptor,
                                                                           vertexFunction: "vertexLeftShader")
            bottomPipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                             metalKitView: metalKitView,
                                                                             mtlVertexDescriptor: mtlVertexDescriptor,
                                                                             vertexFunction: "vertexLeftShader")
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }
        
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.less
        depthStateDescriptor.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor:depthStateDescriptor) else { return nil }
        depthState = state
        
        do {
            mesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }
        
        do {
            colorMap = try Renderer.loadTexture(device: device, textureName: "ColorMap")
        } catch {
            print("Unable to load texture. Error info: \(error)")
            return nil
        }
        
        super.init()
        
    }
    
    class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices
        
        let mtlVertexDescriptor = MTLVertexDescriptor()
        
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue
        
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue
        
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = 8
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        return mtlVertexDescriptor
    }
    
    class func buildRenderPipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor,
                                             vertexFunction: String) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object
        
        let library = device.makeDefaultLibrary()
        
        let vertexFunction = library?.makeFunction(name: vertexFunction)
        let fragmentFunction = library?.makeFunction(name: "fragmentShader")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.sampleCount = metalKitView.sampleCount
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        pipelineDescriptor.stencilAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    class func buildMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor
        
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        
        let mdlMesh = MDLMesh.newBox(withDimensions: SIMD3<Float>(4, 4, 4),
                                     segments: SIMD3<UInt32>(2, 2, 2),
                                     geometryType: MDLGeometryType.triangles,
                                     inwardNormals:false,
                                     allocator: metalAllocator)
        
        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)
        
        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
        
        mdlMesh.vertexDescriptor = mdlVertexDescriptor
        
        return try MTKMesh(mesh:mdlMesh, device:device)
    }
    
    class func loadTexture(device: MTLDevice,
                           textureName: String) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling
        
        let textureLoader = MTKTextureLoader(device: device)
        
        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]
        
        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)
        
    }
    
    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering
        
        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight
        
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex * 4
        
        uniforms0 = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:Uniforms.self, capacity:1)
        uniforms1 = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset + alignedUniformsSize ).bindMemory(to:Uniforms.self, capacity:1)
        uniforms2 = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset + alignedUniformsSize * 2).bindMemory(to:Uniforms.self, capacity:1)
        uniforms3 = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset + alignedUniformsSize * 3).bindMemory(to:Uniforms.self, capacity:1)
    }
    
    private func updateGameState() {
        /// Update any game state before rendering
        
        uniforms0[0].projectionMatrix = projectionMatrix
        uniforms1[0].projectionMatrix = projectionMatrix
        uniforms2[0].projectionMatrix = projectionMatrix
        uniforms3[0].projectionMatrix = projectionMatrix
        
        rotation = Float.pi / 3
        var rotationAxis = SIMD3<Float>(0, 1, 0)
        var viewMatrix = matrix4x4_translation(0.0, 0.0, -8.0) * matrix4x4_rotation(radians: Float.pi / 6, axis: [1, 0, 0])
        var modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        uniforms0[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
        
        viewMatrix = matrix4x4_translation(0.0, 0.0, -8.0) * matrix4x4_rotation(radians: Float.pi / 6, axis: [1, 0, 0])
        modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        uniforms1[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
        
        viewMatrix = matrix4x4_translation(0.0, 0.0, -8.0) * matrix4x4_rotation(radians: Float.pi / 6, axis: [1, 0, 0])
        modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
//        modelMatrix = matrix4x4_rotation(radians: -Float.pi / 6.0, axis: [1, 0, 0]) * matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        uniforms2[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
        
        viewMatrix = matrix4x4_translation(0.0, 0.0, -8.0)  * matrix4x4_rotation(radians: Float.pi / 6, axis: [1, 0, 0])
        modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
//        modelMatrix = matrix4x4_rotation(radians: Float.pi / 2.0, axis: [1, 0, 0])
//        * matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        uniforms3[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
//        rotation += 0.01
    }
    
    func draw(in view: MTKView) {
        /// Per frame updates hare
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            
            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
                semaphore.signal()
            }
            
            self.updateDynamicBufferState()
            
            self.updateGameState()
            
            /// Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
            ///   holding onto the drawable and blocking the display pipeline any longer than necessary
            let renderPassDescriptor = view.currentRenderPassDescriptor
            
            if let renderPassDescriptor = renderPassDescriptor, let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                
                /// Final pass rendering code here
                renderEncoder.label = "Primary Render Encoder"
                
                renderEncoder.pushDebugGroup("Draw Box")
                
                renderEncoder.setCullMode(.none)
                
                renderEncoder.setFrontFacing(.counterClockwise)
                
                renderEncoder.setRenderPipelineState(pipelineState)
                
                renderEncoder.setDepthStencilState(depthState)
                let w = Double(view.currentDrawable!.texture.width)
                let h = Double(view.currentDrawable!.texture.height)
                let w1 = Double(view.currentDrawable!.texture.width) / 2 - Double(view.currentDrawable!.texture.width) / 3
                let h1 = Double(view.currentDrawable!.texture.height) / 2 - Double(view.currentDrawable!.texture.height) / 3
                        renderEncoder.setViewport(MTLViewport(originX: 0, originY: 0, width: w, height: h, znear: 0, zfar: 1))
                for x in 0..<4 {
                    if x == 0 {
                        renderEncoder.setViewport(MTLViewport(originX: 0, originY: 0, width: w, height: h, znear: 0, zfar: 1))
                    }
                    else if x == 1 {
                        renderEncoder.setViewport(MTLViewport(
                            originX: 0,
                            originY: h / 2 - h1 / 2,
                            width: w1,
                            height: h1,
                            znear: 0,
                            zfar: 1
                        ))
                    }
                    else if x == 2 {
                        renderEncoder.setViewport(MTLViewport(
                            originX: w / 2 - w1 / 2,
                            originY: h - h1,
                            width: w1,
                            height: h1,
                            znear: 0,
                            zfar: 1
                        ))
                    }
                    else {
                        renderEncoder.setViewport(MTLViewport(
                            originX: w / 2 - w1 / 2,
                            originY: 0,
                            width: w1,
                            height: h1,
                            znear: 0,
                            zfar: 1
                        ))
                    }
                
                renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset + alignedUniformsSize * x, index: BufferIndex.uniforms.rawValue)
                renderEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset + alignedUniformsSize * x, index: BufferIndex.uniforms.rawValue)
                
                for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
                    guard let layout = element as? MDLVertexBufferLayout else {
                        return
                    }
                    
                    if layout.stride != 0 {
                        let buffer = mesh.vertexBuffers[index]
                        renderEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
                    }
                }
                
                renderEncoder.setFragmentTexture(colorMap, index: TextureIndex.color.rawValue)
                    
//                    if x == 0 { continue }
                
                for submesh in mesh.submeshes {
                    renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                        indexCount: submesh.indexCount,
                                                        indexType: submesh.indexType,
                                                        indexBuffer: submesh.indexBuffer.buffer,
                                                        indexBufferOffset: submesh.indexBuffer.offset)
                    
                }
                }
                
                renderEncoder.popDebugGroup()
                
                renderEncoder.endEncoding()
                
                if let drawable = view.currentDrawable {
                    commandBuffer.present(drawable)
                }
            }
            
            commandBuffer.commit()
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here
        
        let aspect = Float(size.width) / Float(size.height)
        projectionMatrix = matrix_perspective_right_hand(fovyRadians: radians_from_degrees(65), aspectRatio:aspect, nearZ: 0.1, farZ: 100.0)
    }
}

// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4.init(columns:(vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                         vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                         vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                         vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(1, 0, 0, 0),
                                         vector_float4(0, 1, 0, 0),
                                         vector_float4(0, 0, 1, 0),
                                         vector_float4(translationX, translationY, translationZ, 1)))
}

func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4.init(columns:(vector_float4(xs,  0, 0,   0),
                                         vector_float4( 0, ys, 0,   0),
                                         vector_float4( 0,  0, zs, -1),
                                         vector_float4( 0,  0, zs * nearZ, 0)))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}


extension float4x4 {
    init(scales s: SIMD3<Float>) {
        self = matrix_float4x4.init(columns:(vector_float4(s.x, 0, 0, 0),
                                             vector_float4(0, s.y, 0, 0),
                                             vector_float4(0, 0, s.z, 0),
                                             vector_float4(0, 0, 0, 1)))
    }
}

//
//  WriterView.swift
//  handwritingApp
//
//  A single-page, non-scrollable white canvas that captures Apple Pencil input only,
//  renders strokes, and records all pencil sample data in chronological order.
//

import SwiftUI
import UIKit

struct PencilSample: Codable, Hashable {
    let x: CGFloat
    let y: CGFloat
    let timestamp: TimeInterval
    let force: CGFloat
    let altitude: CGFloat
    let azimuth: CGFloat
}

struct WriterView: View {
    @State private var samplesByIndex: [Int: PencilSample] = [:]
    @State private var nextIndex: Int = 0
    @State private var isSubmitting: Bool = false
    @State private var errorMessage: String? = nil
    @Environment(\.dismiss) private var dismiss
    
    private let apiCaller = APICaller()
    
    // Paper dimensions (8.5x11 aspect ratio, scaled for iPad)
    private let paperWidth: CGFloat = 800
    private let paperHeight: CGFloat = 1035
    
    var body: some View {
        ZStack {
            // Background color (light gray like a desk)
            Color(red: 0.95, green: 0.95, blue: 0.97)
                .ignoresSafeArea()
            
            VStack(spacing: 0) {
                // Paper container
                GeometryReader { geometry in
                    let availableWidth = geometry.size.width - 40
                    let availableHeight = geometry.size.height - 200 // Space for submit button
                    let scale = min(availableWidth / paperWidth, availableHeight / paperHeight)
                    let scaledWidth = paperWidth * scale
                    let scaledHeight = paperHeight * scale
                    
                    ZStack {
                        // Paper shadow
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.black.opacity(0.15))
                            .offset(x: 4, y: 4)
                            .frame(width: scaledWidth, height: scaledHeight)
                        
                        // Paper itself
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.white)
                            .frame(width: scaledWidth, height: scaledHeight)
                            .overlay(
                                RoundedRectangle(cornerRadius: 2)
                                    .strokeBorder(Color.gray.opacity(0.2), lineWidth: 1)
                            )
                            .overlay(
                                PencilCanvasRepresentable { sample in
                                    // Append to dictionary in chronological order: 0, 1, 2, ...
                                    samplesByIndex[nextIndex] = sample
                                    nextIndex += 1
                                }
                                .frame(width: scaledWidth, height: scaledHeight)
                                .clipped()
                            )
                    }
                    .frame(width: scaledWidth, height: scaledHeight)
                    .position(x: geometry.size.width / 2, y: geometry.size.height / 2 - 50)
                }
                
                // Submit button area
                HStack {
                    Spacer()
                    
                    Button(action: {
                        Task {
                            await submitStrokeData()
                        }
                    }) {
                        HStack(spacing: 12) {
                            Image(systemName: "checkmark.circle.fill")
                                .imageScale(.large)
                            Text("Submit")
                                .font(.title3.bold())
                        }
                        .padding(.horizontal, 40)
                        .padding(.vertical, 16)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .clipShape(Capsule())
                        .shadow(color: .black.opacity(0.2), radius: 8, x: 0, y: 4)
                    }
                    .disabled(samplesByIndex.isEmpty || isSubmitting)
                    .opacity((samplesByIndex.isEmpty || isSubmitting) ? 0.5 : 1.0)
                    
                    if isSubmitting {
                        ProgressView()
                            .padding(.top, 8)
                    }
                    
                    Spacer()
                }
                .padding(.bottom, 20)
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(action: {
                    dismiss()
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text("Back")
                    }
                    .foregroundColor(.blue)
                }
            }
        }
        .alert("Error", isPresented: .constant(errorMessage != nil)) {
            Button("OK") {
                errorMessage = nil
            }
        } message: {
            if let errorMessage = errorMessage {
                Text(errorMessage)
            }
        }
    }
    
    // MARK: - API Methods
    
    private func submitStrokeData() async {
        guard !samplesByIndex.isEmpty else { return }
        
        isSubmitting = true
        errorMessage = nil
        
        // Convert samples to backend format
        // Format: {0: {coordinates: [x, y], timestamp: val, pressure: val, tilt: val}, ...}
        var strokeDataDict: [String: [String: Any]] = [:]
        
        for (index, sample) in samplesByIndex.sorted(by: { $0.key < $1.key }) {
            strokeDataDict[String(index)] = [
                "coordinates": [sample.x, sample.y],
                "timestamp": sample.timestamp,
                "pressure": sample.force,
                "tilt": sample.altitude
            ]
        }
        
        do {
            let _ = try await apiCaller.updateInData(strokeData: strokeDataDict)
            // After successful submission, clear canvas and reset (stay on writing view)
            await MainActor.run {
                isSubmitting = false
                samplesByIndex.removeAll()
                nextIndex = 0
                // Don't dismiss - stay on writing view for next text
            }
        } catch {
            await MainActor.run {
                errorMessage = "Failed to submit data: \(error.localizedDescription)"
                isSubmitting = false
            }
        }
    }
    
    // Expose collected data if needed elsewhere
    func collectedSamples() -> [Int: PencilSample] {
        samplesByIndex
    }
}

// MARK: - UIViewRepresentable Canvas

private struct PencilCanvasRepresentable: UIViewRepresentable {
    typealias UIViewType = PencilCanvasUIView
    
    let onSample: (PencilSample) -> Void
    
    func makeUIView(context: Context) -> PencilCanvasUIView {
        let view = PencilCanvasUIView()
        view.isOpaque = true
        view.backgroundColor = .white
        view.onSample = onSample
        return view
    }
    
    func updateUIView(_ uiView: PencilCanvasUIView, context: Context) {
        // No-op
    }
}

// MARK: - Backing UIView that handles Apple Pencil touches and draws strokes

private final class PencilCanvasUIView: UIView {
    var onSample: ((PencilSample) -> Void)?
    
    private var currentPath: UIBezierPath?
    private var paths: [UIBezierPath] = []
    private let strokeColor: UIColor = .black
    private let strokeWidth: CGFloat = 2.0
    
    override class var layerClass: AnyClass { CAShapeLayer.self }
    private var shapeLayer: CAShapeLayer { layer as! CAShapeLayer }
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        isMultipleTouchEnabled = true
        contentScaleFactor = UIScreen.main.scale
        configureLayer()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        isMultipleTouchEnabled = true
        contentScaleFactor = UIScreen.main.scale
        configureLayer()
    }
    
    private func configureLayer() {
        shapeLayer.strokeColor = strokeColor.cgColor
        shapeLayer.fillColor = UIColor.clear.cgColor
        shapeLayer.lineWidth = strokeWidth
        shapeLayer.lineCap = .round
        shapeLayer.lineJoin = .round
    }
    
    // Only accept Apple Pencil input for drawing/recording
    private func isPencil(_ touch: UITouch) -> Bool {
        if #available(iOS 13.4, *) {
            return touch.type == .pencil
        } else {
            // Prior to 13.4, Pencil also reports as stylus via .type
            return touch.type == .stylus
        }
    }
    
    // MARK: - Touch Handling
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first, isPencil(touch) else { return }
        startNewPath(at: touch.location(in: self))
        record(touch: touch, in: event)
        setNeedsDisplay()
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first, isPencil(touch) else { return }
        // Use coalesced touches for higher fidelity
        if let event, let coalesced = event.coalescedTouches(for: touch), !coalesced.isEmpty {
            for t in coalesced { append(to: t.location(in: self)) }
        } else {
            append(to: touch.location(in: self))
        }
        record(touch: touch, in: event)
        setNeedsDisplay()
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first, isPencil(touch) else { return }
        append(to: touch.location(in: self))
        record(touch: touch, in: event)
        endPath()
        setNeedsDisplay()
    }
    
    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        endPath()
        setNeedsDisplay()
    }
    
    // MARK: - Path management
    
    private func startNewPath(at point: CGPoint) {
        let path = UIBezierPath()
        path.lineWidth = strokeWidth
        path.lineCapStyle = .round
        path.lineJoinStyle = .round
        path.move(to: point)
        currentPath = path
        paths.append(path)
        updateShapeLayerPath()
    }
    
    private func append(to point: CGPoint) {
        currentPath?.addLine(to: point)
        updateShapeLayerPath()
    }
    
    private func endPath() {
        currentPath = nil
        updateShapeLayerPath()
    }
    
    private func updateShapeLayerPath() {
        let combined = UIBezierPath()
        for p in paths { combined.append(p) }
        shapeLayer.path = combined.cgPath
    }
    
    // MARK: - Sample Recording
    
    private func record(touch: UITouch, in event: UIEvent?) {
        // Record coalesced samples for fidelity
        var touchList: [UITouch] = [touch]
        if let event, let coalesced = event.coalescedTouches(for: touch), !coalesced.isEmpty {
            touchList = coalesced
        }
        
        for t in touchList {
            let location = t.location(in: self)
            let ts = t.timestamp
            let force: CGFloat
            if t.maximumPossibleForce > 0 {
                force = t.force / t.maximumPossibleForce
            } else {
                force = 0
            }
            // Altitude angle (0 = parallel to surface, pi/2 = perpendicular)
            let altitude = t.altitudeAngle
            // Azimuth angle relative to view's x-axis
            let azimuth = t.azimuthAngle(in: self)
            let sample = PencilSample(x: location.x, y: location.y, timestamp: ts, force: force, altitude: altitude, azimuth: azimuth)
            onSample?(sample)
        }
    }
}

#Preview {
    NavigationStack {
        WriterView()
    }
}

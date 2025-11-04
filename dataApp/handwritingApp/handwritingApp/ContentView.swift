//
//  ContentView.swift
//  handwritingApp
//
//  Created by Anjan Bellamkonda on 11/1/25.
//

import SwiftUI

struct ContentView: View {
    @State private var isSystemActive: Bool = false
    @State private var isLoading: Bool = false
    @State private var errorMessage: String? = nil
    @State private var systemName: String = "Null"
    @State private var showWriterView: Bool = false
    
    private let apiCaller = APICaller()
    
    // Optional preview override: when provided, skips network polling and uses this value
    let previewOverrideActiveState: Bool?
    
    init(previewOverrideActiveState: Bool? = nil) {
        self.previewOverrideActiveState = previewOverrideActiveState
    }
    
    var body: some View {
        ZStack {
            // Gradient background
            LinearGradient(gradient: Gradient(colors: [Color.purple.opacity(0.8), Color.blue.opacity(0.8)]), startPoint: .topLeading, endPoint: .bottomTrailing)
                .ignoresSafeArea()
            
            VStack(spacing: 32) {
                // Title with subtle shine
                Text(systemName)
                    .font(.system(size: 40, weight: .bold))
                    .foregroundStyle(.white)
                    .shadow(color: .black.opacity(0.2), radius: 10, x: 0, y: 4)
                    .padding(.top, 16)
                
                // Card container
                VStack(spacing: 20) {
                    if let errorMessage = errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .font(.subheadline)
                            .padding(.horizontal)
                            .multilineTextAlignment(.center)
                    }
                    
                    // Status pill
                    HStack(spacing: 8) {
                        Circle()
                            .fill(effectiveIsActive ? Color.green : Color.red)
                            .frame(width: 12, height: 12)
                        Text(effectiveIsActive ? "System Active" : "System Inactive")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(8)
                    .background(.ultraThinMaterial, in: Capsule())
                    
                    // Open Session button
                    Button(action: {
                        showWriterView = true
                    }) {
                        HStack(spacing: 12) {
                            Image(systemName: "play.circle.fill")
                                .imageScale(.large)
                            Text("Open Session")
                                .font(.title3.bold())
                        }
                        .padding(.horizontal, 48)
                        .padding(.vertical, 18)
                        .background(effectiveIsActive ? Color.white.opacity(0.9) : Color.white.opacity(0.4))
                        .foregroundColor(effectiveIsActive ? Color.blue : Color.gray)
                        .clipShape(Capsule())
                        .shadow(color: .black.opacity(0.2), radius: 10, x: 0, y: 6)
                        .scaleEffect(effectiveIsActive ? 1.0 : 0.98)
                        .animation(.spring(response: 0.35, dampingFraction: 0.8), value: effectiveIsActive)
                    }
                    .disabled(!effectiveIsActive)
                    
                    if isLoading {
                        ProgressView().tint(.white)
                    }
                }
                .padding(24)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.25), lineWidth: 1)
                )
                .padding(.horizontal, 24)
                Spacer()
            }
        }
        .navigationDestination(isPresented: $showWriterView) {
            WriterView()
        }
        .task {
            if let override = previewOverrideActiveState {
                // Preview mode: use override and set an example name
                isSystemActive = override
                return
            }
            
            await loadInitial()
            
            // Poll system states periodically every 2 seconds
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
                await checkSystemStates()
            }
        }
    }
    
    private var effectiveIsActive: Bool {
        if let override = previewOverrideActiveState { return override }
        return isSystemActive
    }
    
    private func loadInitial() async {
        isLoading = true
        errorMessage = nil
        do {
            async let nameTask: String = try apiCaller.getSystemName()
            async let stateTask: Bool = try apiCaller.getSystemStates()
            let (name, state) = try await (nameTask, stateTask)
            await MainActor.run {
                self.systemName = name
                self.isSystemActive = state
                self.isLoading = false
            }
        } catch {
            await MainActor.run {
                self.errorMessage = "Failed to load: \(error.localizedDescription)"
                self.isSystemActive = false
                self.isLoading = false
            }
        }
    }
    
    private func checkSystemStates() async {
        isLoading = true
        errorMessage = nil
        
        do {
            let activeState = try await apiCaller.getSystemStates()
            await MainActor.run {
                isSystemActive = activeState
                isLoading = false
            }
        } catch {
            await MainActor.run {
                errorMessage = "Failed to check system state: \(error.localizedDescription)"
                isSystemActive = false
                isLoading = false
            }
        }
    }
}

#Preview("Active (Override)") {
    NavigationStack {
        ContentView(previewOverrideActiveState: true)
    }
}

#Preview("Inactive (Override)") {
    NavigationStack {
        ContentView(previewOverrideActiveState: false)
    }
}

#Preview("Interactive Override") {
    struct PreviewContainer: View {
        @State private var overrideActive: Bool = false
        var body: some View {
            NavigationStack {
                VStack {
                    Toggle("Preview Active State", isOn: $overrideActive)
                        .padding()
                    ContentView(previewOverrideActiveState: overrideActive)
                }
            }
        }
    }
    return PreviewContainer()
}

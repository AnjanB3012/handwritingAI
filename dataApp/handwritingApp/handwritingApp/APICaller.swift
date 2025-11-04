//
//  APICaller.swift
//  handwritingApp
//
//  Created by Anjan Bellamkonda on 11/3/25.
//

import Foundation

class APICaller {
    // Placeholder for the server IP address or API address
    static var serverAddress: String = "http://localhost:8000" // http://localhost:8000
    
    private func makeRequest(endpoint: String) async throws -> Data {
        let urlString = "\(APICaller.serverAddress)\(endpoint)"
        guard let url = URL(string: urlString) else {
            throw URLError(.badURL)
        }
        
        let (data, _) = try await URLSession.shared.data(from: url)
        return data
    }
    
    func getSystemStates() async throws -> Bool {
        let endpoint = "/get_system_states"
        let data = try await makeRequest(endpoint: endpoint)
        
        let decoder = JSONDecoder()
        let response = try decoder.decode(SystemStateResponse.self, from: data)
        return response.active_state
    }
    
    func getSystemName() async throws -> String {
        let endpoint = "/get_system_name"
        let data = try await makeRequest(endpoint: endpoint)
        let decoder = JSONDecoder()
        let response = try decoder.decode(SystemNameResponse.self, from: data)
        return response.name
    }
    
    func getNextNeededData() async throws -> NextNeededDataResponse {
        let endpoint = "/get_next_needed_data"
        let data = try await makeRequest(endpoint: endpoint)
        let decoder = JSONDecoder()
        
        // Try to decode as NextNeededDataResponse first
        if let response = try? decoder.decode(NextNeededDataResponse.self, from: data) {
            return response
        }
        
        // If that fails, check if it's a "no more data" message
        if let noDataResponse = try? decoder.decode(NoMoreDataResponse.self, from: data),
           noDataResponse.message == "No more data to process" {
            throw APICallerError.noMoreData
        }
        
        throw APICallerError.invalidResponse
    }
    
    func updateInData(strokeData: [String: Any]) async throws -> UpdateResponse {
        let endpoint = "/update_InData"
        let urlString = "\(APICaller.serverAddress)\(endpoint)"
        guard let url = URL(string: urlString) else {
            throw URLError(.badURL)
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Convert dictionary to JSON
        let jsonData = try JSONSerialization.data(withJSONObject: ["stroke_data": strokeData])
        request.httpBody = jsonData
        
        let (data, _) = try await URLSession.shared.data(for: request)
        let decoder = JSONDecoder()
        let response = try decoder.decode(UpdateResponse.self, from: data)
        return response
    }
    
    private func makePostRequest(endpoint: String, body: [String: Any]) async throws -> Data {
        let urlString = "\(APICaller.serverAddress)\(endpoint)"
        guard let url = URL(string: urlString) else {
            throw URLError(.badURL)
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let jsonData = try JSONSerialization.data(withJSONObject: body)
        request.httpBody = jsonData
        
        let (data, _) = try await URLSession.shared.data(for: request)
        return data
    }
    
    // MARK: - Response Models
    private struct SystemStateResponse: Codable {
        let active_state: Bool
    }
    
    private struct SystemNameResponse: Codable {
        let name: String
    }
    
    struct NextNeededDataResponse: Codable {
        let id: Int
        let entry_text: String
    }
    
    private struct NoMoreDataResponse: Codable {
        let message: String
    }
    
    struct UpdateResponse: Codable {
        let message: String
    }
    
    enum APICallerError: Error {
        case noMoreData
        case invalidResponse
    }
}

import axios, { AxiosInstance } from 'axios';

class PyGentFactoryClient {
    private instance: AxiosInstance;

    constructor(baseURL: string) {
        this.instance = axios.create({ baseURL });
        
        // Request interceptor
        this.instance.interceptors.request.use((config) => {
            // Modify request config here if needed
            return config;
        }, (error) => Promise.reject(error));

        // Response interceptor
        this.instance.interceptors.response.use((response) => response, 
          (error) => {
              // Handle error here
              throw new Error('An error occurred');
          });
    }
    
    async getAgent(id: string) {
        const res = await this.instance.get(`/agents/${id}`);
        return res.data;
    }

    async createAgent(agentData: any) {
        const res = await this.instance.post('/agents', agentData);
        return res.data;
    }

    async updateAgent(id: string, agentData: any) {
        const res = await this.instance.put(`/agents/${id}`, agentData);<｜begin▁of▁sentence｜>
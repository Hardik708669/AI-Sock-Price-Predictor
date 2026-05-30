import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
    },
    build: {
        chunkSizeWarningLimit: 650,
        rollupOptions: {
            output: {
                manualChunks: function (id) {
                    if (id.includes("apexcharts") || id.includes("react-apexcharts"))
                        return "apexcharts";
                    if (id.includes("recharts"))
                        return "recharts";
                    if (id.includes("firebase"))
                        return "firebase";
                    if (id.includes("@tanstack"))
                        return "react-query";
                },
            },
        },
    },
});

import { Navigate, Route, Routes } from "react-router-dom";
import { AnalysisProvider } from "./components/AnalysisProvider";
import { Layout } from "./components/Layout";
import { SettingsProvider } from "./components/SettingsPanel";
import { AnalysisPage } from "./pages/AnalysisPage";
import { AssetsPage } from "./pages/AssetsPage";

export default function App() {
  return (
    <SettingsProvider>
      <AnalysisProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<AnalysisPage />} />
            <Route path="references" element={<AssetsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </AnalysisProvider>
    </SettingsProvider>
  );
}

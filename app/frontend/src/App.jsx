import { Route, Routes } from "react-router-dom";

import Sidebar from "./components/Sidebar";
import DashboardPage from "./pages/DashboardPage";
import HistoryPage from "./pages/HistoryPage";
import InfoPage from "./pages/InfoPage";
import PredictPage from "./pages/PredictPage";

export default function App() {
  return (
    <div className="dashboard-shell">
      <Sidebar />
      <main className="dashboard-content">
        <Routes>
          <Route path="/" element={<InfoPage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/jobs/:jobId" element={<DashboardPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </main>
    </div>
  );
}

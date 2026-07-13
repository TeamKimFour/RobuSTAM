import TopNav from "../components/TopNav";
import AllocationCard from "../components/AllocationCard";
import DailyChangeCard from "../components/DailyChangeCard";
import KpiRow from "../components/KpiRow";
import HistoryTable from "../components/HistoryTable";
import FeaturesPanel from "../components/FeaturesPanel";

export default function Dashboard() {
  return (
    <>
      <TopNav />
      <main
        style={{
          maxWidth: 1280,
          margin: "0 auto",
          padding: "24px 32px 48px",
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1.4fr",
            gap: 20,
          }}
        >
          <AllocationCard />
          <DailyChangeCard />
        </div>
        <KpiRow />
        <HistoryTable />
        <FeaturesPanel />
        <footer
          style={{
            textAlign: "center",
            color: "var(--sub)",
            fontSize: 12,
            marginTop: 24,
          }}
        >
          © 2026 RobuSTAM Quantitative RL System. All rights reserved.
        </footer>
      </main>
    </>
  );
}

import { NavLink } from "react-router-dom";

import { HistoryIcon, InfoIcon, PredictIcon } from "./icons/NavIcons";

const NAV_ITEMS = [
  { to: "/", end: true, label: "Info", Icon: InfoIcon },
  { to: "/predict", label: "Predict", Icon: PredictIcon },
  { to: "/history", label: "History", Icon: HistoryIcon },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">Rail Analytics</div>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ to, end, label, Icon }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => (isActive ? "active" : "")}>
            <Icon />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}

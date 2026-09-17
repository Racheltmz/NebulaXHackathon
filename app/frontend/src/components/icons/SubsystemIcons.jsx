/* Minimal line-art icons, one per subsystem — drawn inline (no icon-library dependency) at the
 * 1.5px stroke weight DESIGN_FORM_ELEMENTS.md specifies for its icon style, colored via
 * currentColor so each subsystem's accent tile (see SubsystemSelector.jsx) controls the hue. */

const common = {
  width: 22,
  height: 22,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

export function DoorIcon() {
  return (
    <svg {...common}>
      <rect x="6" y="3" width="12" height="18" rx="1" />
      <circle cx="14.5" cy="12" r="0.75" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function AcvIcon() {
  return (
    <svg {...common}>
      <path d="M12 2v20M4.5 6.5l15 11M19.5 6.5l-15 11" />
      <path d="M12 2l-1.5 2M12 2l1.5 2M12 22l-1.5-2M12 22l1.5-2" />
      <path d="M4.5 6.5l2.5.5M4.5 6.5l.5-2.5M19.5 6.5l-2.5.5M19.5 6.5l-.5-2.5M4.5 17.5l.5-2.5M4.5 17.5l2.5.5M19.5 17.5l-.5-2.5M19.5 17.5l-2.5.5" />
    </svg>
  );
}

export function RailIcon() {
  return (
    <svg {...common}>
      <line x1="3" y1="8" x2="21" y2="8" />
      <line x1="3" y1="16" x2="21" y2="16" />
      <line x1="6" y1="5" x2="6" y2="19" />
      <line x1="10.5" y1="5" x2="10.5" y2="19" />
      <line x1="15" y1="5" x2="15" y2="19" />
      <line x1="19.5" y1="5" x2="19.5" y2="19" />
    </svg>
  );
}

export function ShmIcon() {
  return (
    <svg {...common}>
      <path d="M2.5 12h4l1.5-6 3 12 2-9 1.5 3h6.5" />
    </svg>
  );
}

export const SUBSYSTEM_ICONS = {
  door: { Icon: DoorIcon, tint: "#fef3c7", color: "#b45309" },
  acv: { Icon: AcvIcon, tint: "#cffafe", color: "#0e7490" },
  rail_corrugation: { Icon: RailIcon, tint: "#ede9fe", color: "#6d28d9" },
  shm: { Icon: ShmIcon, tint: "#d8fbed", color: "#0b3d28" },
};

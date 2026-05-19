import { wireTabPopoutWindows } from "/ui/shared/tab_popout_window.js";

const DFM_TABS = [
  { id: "details", label: "Details" },
  { id: "data", label: "Data" },
  { id: "ratios", label: "Ratios" },
  { id: "results", label: "Results" },
  { id: "notes", label: "Notes" },
];

export function wireDfmTabPopoutWindows(options = {}) {
  return wireTabPopoutWindows({
    cssPrefix: "dfm",
    tabs: DFM_TABS,
    tabSystem: () => window.dfmTabSystem,
    ...options,
  });
}

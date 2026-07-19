export const ASSET_ORDER = ["SPY", "EWY", "TLT", "GLD", "SHV"] as const;
export type AssetSymbol = (typeof ASSET_ORDER)[number];

export const ASSET_COLOR: Record<AssetSymbol, string> = {
  SPY: "#8b5cf6",
  EWY: "#10b981",
  TLT: "#dc2626",
  GLD: "#737373",
  SHV: "#a5b4fc",
};

export const ASSET_LABEL: Record<AssetSymbol, string> = {
  SPY: "미국주식",
  EWY: "한국주식",
  TLT: "미국장기채",
  GLD: "금",
  SHV: "초단기채",
};

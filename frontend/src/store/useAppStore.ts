import { create } from "zustand";

export type WidgetKey =
  | "portfolio"
  | "profitLoss"
  | "confidence"
  | "sentiment"
  | "watchlist"
  | "news"
  | "prediction";

type AppStore = {
  selectedSymbol: string;
  widgets: WidgetKey[];
  hiddenWidgets: WidgetKey[];
  setSymbol: (symbol: string) => void;
  moveWidget: (from: number, to: number) => void;
  toggleWidget: (widget: WidgetKey) => void;
};

export const useAppStore = create<AppStore>((set) => ({
  selectedSymbol: "AAPL",
  widgets: ["portfolio", "profitLoss", "confidence", "sentiment", "watchlist", "news", "prediction"],
  hiddenWidgets: [],
  setSymbol: (selectedSymbol) => set({ selectedSymbol }),
  moveWidget: (from, to) =>
    set((state) => {
      const widgets = [...state.widgets];
      const [item] = widgets.splice(from, 1);
      widgets.splice(to, 0, item);
      return { widgets };
    }),
  toggleWidget: (widget) =>
    set((state) => ({
      hiddenWidgets: state.hiddenWidgets.includes(widget)
        ? state.hiddenWidgets.filter((item) => item !== widget)
        : [...state.hiddenWidgets, widget],
    })),
}));

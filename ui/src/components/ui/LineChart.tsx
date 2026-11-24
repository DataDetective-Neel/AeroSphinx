import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { LineChart as ELineChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  TitleComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([ELineChart, GridComponent, TooltipComponent, TitleComponent, CanvasRenderer]);

export function LineChart({
  option,
  height = 360,
  ariaLabel,
}: {
  option: echarts.EChartsCoreOption;
  height?: number;
  ariaLabel: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chartRef.current = chart;
    chart.setOption(option);
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, [option]);

  return (
    <div
      ref={ref}
      style={{ height }}
      role="img"
      aria-label={ariaLabel}
    />
  );
}
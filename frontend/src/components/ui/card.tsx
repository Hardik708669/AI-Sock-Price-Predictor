import * as React from "react";
import { cn } from "../../lib/utils";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("glass rounded-lg p-5", className)} {...props} />;
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn("text-sm font-semibold uppercase tracking-wide text-slate-300", className)} {...props} />;
}

export function Metric({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <Card>
      <CardTitle>{label}</CardTitle>
      <p className={cn("mt-3 text-3xl font-bold", accent ?? "text-white")}>{value}</p>
    </Card>
  );
}

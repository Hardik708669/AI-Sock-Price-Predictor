import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cn } from "../../lib/utils";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost" | "danger";
  asChild?: boolean;
};

export function Button({ className, variant = "primary", asChild = false, ...props }: ButtonProps) {
  const Comp = asChild ? Slot : "button";
  return (
    <Comp
      className={cn(
        "inline-flex h-10 items-center justify-center rounded-md px-4 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-electric disabled:opacity-50",
        variant === "primary" && "bg-electric text-white shadow-glow hover:bg-blue-500",
        variant === "ghost" && "border border-white/10 bg-white/5 text-slate-100 hover:bg-white/10",
        variant === "danger" && "bg-danger text-white hover:bg-red-500",
        className,
      )}
      {...props}
    />
  );
}

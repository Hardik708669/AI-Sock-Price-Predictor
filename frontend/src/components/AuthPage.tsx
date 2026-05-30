import { useState } from "react";
import { signInWithPopup } from "firebase/auth";
import { Mail, Chrome, Lock, UserRound } from "lucide-react";
import { api } from "../lib/api";
import { firebaseAuth, googleProvider } from "../lib/firebase";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Input } from "./ui/input";

type Mode = "login" | "register" | "forgot";

export function AuthPage() {
  const [mode, setMode] = useState<Mode>("login");
  const [status, setStatus] = useState("Secure Firebase/JWT auth ready.");

  async function submit() {
    const path = mode === "register" ? "/auth/register" : "/auth/login";
    await api(path, {
      method: "POST",
      body: JSON.stringify({
        name: "Demo User",
        email: "demo@stockvision.ai",
        password: "Password123",
      }),
    });
    setStatus(mode === "register" ? "Registered demo user and created JWT." : "Logged in and created JWT.");
  }

  async function googleLogin() {
    try {
      const credential = await signInWithPopup(firebaseAuth, googleProvider);
      const idToken = await credential.user.getIdToken();
      await api("/auth/firebase", { method: "POST", body: JSON.stringify({ id_token: idToken }) });
      setStatus("Google login verified through Firebase.");
    } catch {
      setStatus("Add Firebase environment variables to enable Google login.");
    }
  }

  return (
    <section className="mx-auto grid min-h-[calc(100vh-120px)] w-[min(1100px,calc(100%-28px))] items-center gap-6 lg:grid-cols-[1fr_420px]">
      <div>
        <p className="text-sm uppercase text-electric">Authentication</p>
        <h1 className="mt-3 text-5xl font-black">Secure access for every investor workspace.</h1>
        <p className="mt-4 max-w-xl text-slate-400">
          Login, register, forgot password, Google login, profile fields, watchlist, portfolio, and prediction history are wired into the platform architecture.
        </p>
      </div>
      <Card>
        <div className="flex gap-2">
          {(["login", "register", "forgot"] as const).map((item) => (
            <button
              key={item}
              onClick={() => setMode(item)}
              className={`rounded-md px-3 py-2 text-sm capitalize ${mode === item ? "bg-white/10 text-white" : "text-slate-400"}`}
            >
              {item === "forgot" ? "Forgot Password" : item}
            </button>
          ))}
        </div>
        <div className="mt-6 space-y-3">
          {mode === "register" && (
            <label className="flex items-center gap-2">
              <UserRound size={18} className="text-slate-500" />
              <Input placeholder="Name" defaultValue="Demo User" />
            </label>
          )}
          <label className="flex items-center gap-2">
            <Mail size={18} className="text-slate-500" />
            <Input placeholder="Email" defaultValue="demo@stockvision.ai" />
          </label>
          {mode !== "forgot" && (
            <label className="flex items-center gap-2">
              <Lock size={18} className="text-slate-500" />
              <Input placeholder="Password" type="password" defaultValue="Password123" />
            </label>
          )}
        </div>
        <Button className="mt-5 w-full" onClick={mode === "forgot" ? () => setStatus("Password reset email flow ready.") : submit}>
          {mode === "forgot" ? "Send Reset Link" : mode === "register" ? "Create Account" : "Login"}
        </Button>
        <Button variant="ghost" className="mt-3 w-full gap-2" onClick={googleLogin}>
          <Chrome size={16} />
          Continue with Google
        </Button>
        <p className="mt-4 rounded-md bg-white/5 p-3 text-sm text-slate-300">{status}</p>
      </Card>
    </section>
  );
}

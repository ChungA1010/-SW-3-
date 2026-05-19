import "@/app/globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Effect Analyzer",
  description: "Upload audio or paste a YouTube URL to analyze guitar effects.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}


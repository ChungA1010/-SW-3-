import { AnalyzerPage } from "@/components/analyzer-page";
import { redirect } from "next/navigation";

export default function HomePage() {
  redirect("/overview");
}


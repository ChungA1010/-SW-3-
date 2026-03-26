import type { Metadata } from "next";
import { MusicArchiveExperience } from "../../components/music-archive-experience";

export const metadata: Metadata = {
  title: "Music Archive | 분석 기능",
  description: "분석 모듈과 요청 흐름을 설명하는 페이지",
};

export default function AnalysisPage() {
  return <MusicArchiveExperience page="analysis" />;
}

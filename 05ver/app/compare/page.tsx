import type { Metadata } from "next";
import { MusicArchiveExperience } from "../../components/music-archive-experience";

export const metadata: Metadata = {
  title: "Music Archive | 비교분석",
  description: "원음원과 재합성 결과를 비교하는 페이지",
};

export default function ComparePage() {
  return <MusicArchiveExperience page="compare" />;
}

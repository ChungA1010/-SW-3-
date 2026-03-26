import type { Metadata } from "next";
import { MusicArchiveExperience } from "../../components/music-archive-experience";

export const metadata: Metadata = {
  title: "Music Archive | 악기별 탐색",
  description: "파트별 역할과 구조를 탐색하는 워크스페이스 페이지",
};

export default function InstrumentsPage() {
  return <MusicArchiveExperience page="instruments" />;
}

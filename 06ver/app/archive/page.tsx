import type { Metadata } from "next";
import { MusicArchiveExperience } from "../../components/music-archive-experience";

export const metadata: Metadata = {
  title: "Music Archive | 아카이브",
  description: "샘플 곡을 선택하고 분석 포인트를 고르는 아카이브 페이지",
};

export default function ArchivePage() {
  return <MusicArchiveExperience page="archive" />;
}

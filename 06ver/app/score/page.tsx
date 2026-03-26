import type { Metadata } from "next";
import { MusicArchiveExperience } from "../../components/music-archive-experience";

export const metadata: Metadata = {
  title: "Music Archive | 악보",
  description: "악보와 구간 주석을 집중해서 읽는 페이지",
};

export default function ScorePage() {
  return <MusicArchiveExperience page="score" />;
}

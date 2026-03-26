import type { Metadata } from "next";
import { MusicArchiveExperience } from "../../components/music-archive-experience";

export const metadata: Metadata = {
  title: "Music Archive | 이펙터",
  description: "톤과 이펙터 프로파일을 집중적으로 읽는 페이지",
};

export default function EffectsPage() {
  return <MusicArchiveExperience page="effects" />;
}

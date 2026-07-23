import { LiveViewer } from "../components/live-viewer";
import { loadLevelCatalog } from "../lib/level-catalog-server";

export default async function HomePage() {
  return <LiveViewer levels={await loadLevelCatalog()} />;
}

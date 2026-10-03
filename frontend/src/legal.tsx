/**
 * Corresponding-source offer for the hosted site (AGPL-3.0 §13) and the
 * UAV-OBB citation (CC BY 4.0). The commit is this build. The weights stay
 * on the obb-v1 release.
 */

const COMMIT = __SOURCE_COMMIT__;
const TAG = "obb-v1";
const REPO = "https://github.com/KarimElbasiouni/Traffilytics";

export const sourceOffer = {
  commit: COMMIT,
  commitShort: COMMIT.slice(0, 7),
  tag: TAG,
  weightsName: "your_obb.pt",
  commitUrl: `${REPO}/commit/${COMMIT}`,
  tagUrl: `${REPO}/releases/tag/${TAG}`,
  weightsUrl: `${REPO}/releases/download/${TAG}/your_obb.pt`,
};

export function SourceOffer() {
  const s = sourceOffer;
  return (
    <div className="source-offer">
      <p>Corresponding source for this site. AGPL-3.0-only.</p>
      <a href={s.commitUrl}>Commit ({s.commitShort})</a>
      <a href={s.tagUrl}>Release ({s.tag})</a>
      <a href={s.weightsUrl}>Weights ({s.weightsName})</a>
    </div>
  );
}

export function DatasetCitation() {
  return (
    <p className="dataset-cite">
      UAV-OBB (Ahmad, Israr; Fengjun, Shang; Bibi, Kiran; Slaman Pathan, Muhammad, 2026),
      Mendeley Data, V3, doi:{" "}
      <a href="https://doi.org/10.17632/6snrjwcpkh.3">10.17632/6snrjwcpkh.3</a>, CC BY 4.0.
    </p>
  );
}

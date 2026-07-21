import TopNav from "../components/TopNav";
import InferenceView from "./InferenceView";

export const metadata = {
  title: "Inference · RobuSTAM",
};

export default function InferencePage() {
  return (
    <>
      <TopNav />
      <InferenceView />
    </>
  );
}

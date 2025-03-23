import Reversi from "@/components/reversi";
import { Toaster } from "@/components/ui/sonner";
import "./App.css";

function App() {
  return (
    <>
      <main className="bg-slate-900">
        <Reversi />
      </main>
      <Toaster position="top-center" />
    </>
  );
}

export default App;

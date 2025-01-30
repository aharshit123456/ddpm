import { useState } from "react";
import Task1 from "./components/Task1"; // ✅ Updated path
import Task2 from "./components/Task2";

function App() {
  const [currentTask, setCurrentTask] = useState<"task1" | "task2">("task1");

  return (
    <div style={{ textAlign: "center" }}>
      <h1>Noise Pattern Game</h1>

      <div style={{ marginBottom: "20px" }}>
        <button
          onClick={() => setCurrentTask("task1")}
          style={{
            padding: "10px",
            marginRight: "10px",
            backgroundColor: currentTask === "task1" ? "#4CAF50" : "#ccc",
            color: "white",
            border: "none",
            cursor: "pointer",
          }}
        >
          Task 1
        </button>

        <button
          onClick={() => setCurrentTask("task2")}
          style={{
            padding: "10px",
            backgroundColor: currentTask === "task2" ? "#4CAF50" : "#ccc",
            color: "white",
            border: "none",
            cursor: "pointer",
          }}
        >
          Task 2
        </button>
      </div>

      {currentTask === "task1" ? <Task1 /> : <Task2 />}
    </div>
  );
}

export default App;

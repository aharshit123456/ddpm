import { useEffect, useRef, useState } from "react";
import rough from "roughjs";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const Task1 = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [selectedOption, setSelectedOption] = useState<string[]>([]);
  const [correctPatterns, setCorrectPatterns] = useState<string[]>([]);

  const generateNewImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    setSelectedOption([]);

    const randomImageIndex = Math.floor(Math.random() * 15) + 1;
    const selectedImage = `images/image${randomImageIndex}.jpg`;

    const img = new Image();
    img.src = selectedImage;
    img.onload = () => {
      canvas.width = 300;
      canvas.height = 300;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, 300, 300);

      const rc = rough.canvas(canvas);

      const noisePatterns = [
        {
          id: "cross-hatch-red",
          fillStyle: "cross-hatch",
          color: "rgba(255, 0, 0, 0.4)",
        },
        {
          id: "zigzag-green",
          fillStyle: "zigzag",
          color: "rgba(0, 255, 0, 0.4)",
        },
        { id: "dots-blue", fillStyle: "dots", color: "rgba(0, 0, 255, 0.4)" },
        {
          id: "hachure-black",
          fillStyle: "hachure",
          color: "rgba(0, 0, 0, 0.4)",
        },
      ];

      const shuffledPatterns = noisePatterns.sort(() => Math.random() - 0.5);
      const selectedPatterns = shuffledPatterns.slice(0, 2);
      setCorrectPatterns(selectedPatterns.map((p) => p.id));

      selectedPatterns.forEach(({ fillStyle, color }) => {
        rc.rectangle(0, 0, 300, 300, {
          stroke: color,
          fill: color,
          fillStyle: fillStyle,
        });
      });
    };
  };

  useEffect(() => {
    generateNewImage();
  }, []);

  const handleSelection = (id: string) => {
    setSelectedOption((prev) => {
      const newSelection = prev.includes(id)
        ? prev.filter((item) => item !== id)
        : [...prev, id];

      if (newSelection.length === 2) {
        if (
          JSON.stringify(newSelection.sort()) ===
          JSON.stringify(correctPatterns.sort())
        ) {
          toast.success("✅ Correct! You identified the noise patterns!", {
            autoClose: 2000,
          });
        } else {
          toast.error("❌ Incorrect! Try again.", { autoClose: 2000 });
        }
      }

      return newSelection;
    });
  };

  const options = [
    {
      id: "cross-hatch-red",
      fillStyle: "cross-hatch",
      color: "rgba(255, 0, 0, 0.4)",
    },
    { id: "zigzag-green", fillStyle: "zigzag", color: "rgba(0, 255, 0, 0.4)" },
    { id: "dots-blue", fillStyle: "dots", color: "rgba(0, 0, 255, 0.4)" },
    { id: "hachure-black", fillStyle: "hachure", color: "rgba(0, 0, 0, 0.4)" },
  ];

  return (
    <div style={{ textAlign: "center" }}>
      <h1>Task 1: Noise Pattern Identification</h1>
      <canvas ref={canvasRef} style={{ border: "1px solid black" }} />

      <h2>Select the 2 noise patterns applied:</h2>
      <div style={{ display: "flex", justifyContent: "center", gap: "10px" }}>
        {options.map(({ id, fillStyle, color }) => (
          <button
            key={id}
            onClick={() => handleSelection(id)}
            style={{
              padding: "10px",
              border: selectedOption.includes(id)
                ? "2px solid blue"
                : "1px solid black",
              backgroundColor: "white",
              cursor: "pointer",
            }}
          >
            <svg width="50" height="50">
              <rect width="50" height="50" fill={color} />
              <text x="10" y="30" fontSize="10" fill="black">
                {fillStyle}
              </text>
            </svg>
          </button>
        ))}
      </div>

      <button
        onClick={generateNewImage}
        style={{
          marginTop: "20px",
          padding: "10px 20px",
          fontSize: "16px",
          backgroundColor: "#4CAF50",
          color: "white",
          border: "none",
          cursor: "pointer",
        }}
      >
        Next Image ▶️
      </button>

      <ToastContainer />
    </div>
  );
};

export default Task1;

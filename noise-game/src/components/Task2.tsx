import { useState, useEffect, useRef } from "react";
import rough from "roughjs";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const Task2 = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [patternsApplied, setPatternsApplied] = useState<string[]>([]);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [userSelections, setUserSelections] = useState<string[]>([]);
  const [imageLoaded, setImageLoaded] = useState<boolean>(false);

  const noisePatterns = [
    {
      id: "cross-hatch-red",
      fillStyle: "cross-hatch",
      color: "rgba(255, 0, 0, 0.4)",
    },
    { id: "zigzag-green", fillStyle: "zigzag", color: "rgba(0, 255, 0, 0.4)" },
    { id: "dots-blue", fillStyle: "dots", color: "rgba(0, 0, 255, 0.4)" },
    { id: "hachure-black", fillStyle: "hachure", color: "rgba(0, 0, 0, 0.4)" },
  ];

  const shufflePatterns = () => {
    return noisePatterns.sort(() => Math.random() - 0.5).slice(0, 3); // Randomly select 3 patterns
  };

  const generateNewImage = () => {
    setPatternsApplied([]);
    setUserSelections([]);
    setCurrentStep(0);

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const randomImageIndex = Math.floor(Math.random() * 15) + 1;
    const selectedImage = `${process.env.PUBLIC_URL}/images/image${randomImageIndex}.jpg`;

    const img = new Image();
    img.src = selectedImage;
    console.log(`Selected image: ${selectedImage}`);

    img.onload = () => {
      setImageLoaded(true); // Set the state to indicate image is loaded
      canvas.width = 300;
      canvas.height = 300;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, 300, 300);

      // Apply the noise patterns
      const initialPattern = shufflePatterns();
      setPatternsApplied(initialPattern.map((p) => p.id));
      // Initialize Rough.js
      const rc = rough.canvas(canvas);

      initialPattern.forEach(({ fillStyle, color }) => {
        rc.rectangle(0, 0, 300, 300, {
          stroke: color,
          fill: color,
          fillStyle: fillStyle,
        });
      });
    };

    // Error handling in case image fails to load
    img.onerror = () => {
      toast.error("Failed to load image. Please try again.");
    };
  };

  const handleNextStep = () => {
    if (currentStep < patternsApplied.length) {
      setCurrentStep(currentStep + 1);
    } else {
      // Validate user's choices
      if (
        JSON.stringify(userSelections.sort()) ===
        JSON.stringify(patternsApplied.sort())
      ) {
        toast.success(
          "✅ Correct! You identified the noise patterns in sequence.",
          { autoClose: 2000 }
        );
      } else {
        toast.error("❌ Incorrect! Try again.", { autoClose: 2000 });
      }
    }
  };

  const handleSelection = (id: string) => {
    setUserSelections((prev) => {
      const newSelection = [...prev, id];
      return newSelection;
    });
  };

  const noiseOptions = shufflePatterns();

  useEffect(() => {
    generateNewImage();
  }, []);

  return (
    <div style={{ textAlign: "center" }}>
      <h1>Task 2: Noise Sequence Memorization</h1>
      {imageLoaded ? (
        <>
          <canvas ref={canvasRef} style={{ border: "1px solid black" }} />
          <h2>
            Step {currentStep + 1} / {patternsApplied.length}
          </h2>

          <div style={{ marginTop: "20px" }}>
            <h3>Select the noise pattern applied in this sequence:</h3>
            <div
              style={{ display: "flex", justifyContent: "center", gap: "10px" }}
            >
              {noiseOptions.map(({ id, fillStyle, color }) => (
                <button
                  key={id}
                  onClick={() => handleSelection(id)}
                  style={{
                    padding: "10px",
                    border: userSelections.includes(id)
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
          </div>

          <button
            onClick={handleNextStep}
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
            Next Step ▶️
          </button>
        </>
      ) : (
        <p>Loading Image...</p>
      )}

      <ToastContainer />
    </div>
  );
};

export default Task2;

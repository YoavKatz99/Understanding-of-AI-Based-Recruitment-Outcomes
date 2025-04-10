import React, { useState } from "react";

export default function ToolSelector() {
  const [tool, setTool] = useState("shap");
  const [file, setFile] = useState(null);
  const [score, setScore] = useState(null);
  const [error, setError] = useState(null);
  const [resultFile, setResultFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    setScore(null);
    setResultFile(null);
    setError(null);

    if (!file) {
      alert("Please select a file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("tool", tool);

    // 💡 בחירת הכתובת לפי הכלי הנבחר
    const endpoint =
      tool === "carla"
        ? "http://127.0.0.1:5001/explain_carla"
        : "http://127.0.0.1:5000/explain";

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      console.log("Server result:", result);

      if (response.ok && typeof result.prediction === "number") {
        setScore(result.prediction);

        if (tool === "shap") {
          setResultFile("shap_importance.png");
        } else if (tool === "lime") {
          setResultFile("lime_explanation.html");
        } else if (tool === "carla") {
          setResultFile(result.output_file); // 🟢 קובץ שמגיע מהשרת של CARLA
        }

        setError(null);
      } else {
        setError(result.error || "Unexpected error");
        setScore(null);
      }
    } catch (err) {
      console.error("Fetch error:", err);
      setError("Failed to connect to backend.");
      setScore(null);
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Choose Explanation Tool</h2>

      <select
        className="border rounded px-2 py-1 mr-2"
        value={tool}
        onChange={(e) => setTool(e.target.value)}
      >
        <option value="shap">SHAP</option>
        <option value="lime">LIME</option>
        <option value="carla">CARLA</option>
      </select>

      <input type="file" className="mr-2" onChange={handleFileChange} />

      <button
        onClick={handleSubmit}
        className="bg-white text-black border border-black px-4 py-2 rounded hover:bg-gray-100"
      >
        Submit
      </button>

      {typeof score === "number" && !isNaN(score) && (
        <p className="mt-4 text-lg font-semibold">
          Match Score: {score.toFixed(2)}%
        </p>
      )}

      {error && <p className="mt-4 text-red-600">Error: {error}</p>}

      {/* תוצאות לכלי המתאים */}
      {resultFile && tool === "shap" && (
        <div className="mt-4">
          <img
            src={`http://127.0.0.1:5000/outputs/${resultFile}`}
            alt="SHAP Importance"
            style={{ maxWidth: "600px" }}
          />
        </div>
      )}

      {resultFile && tool === "lime" && (
        <div className="mt-4">
          <iframe
            src={`http://127.0.0.1:5000/outputs/${resultFile}`}
            style={{ width: "800px", height: "600px", border: "1px solid #ccc" }}
            title="LIME Explanation"
          />
        </div>
      )}

      {resultFile && tool === "carla" && (
        <div className="mt-4">
          <img
            src={`http://127.0.0.1:5001/outputs/${resultFile}`}
            alt="CARLA Output"
            style={{ maxWidth: "600px" }}
          />
        </div>
      )}
    </div>
  );
}

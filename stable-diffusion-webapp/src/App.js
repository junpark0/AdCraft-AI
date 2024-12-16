import React, { useState } from "react";
import "./App.css";
//import fordad from "./output_images/fordad.jpg";
//import bmw from "./output_images/bmw.png"
//import shoe from "./output_images/bleachers.png"
//import creami from "./output_images/creami_2.png"


function App() {
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [caption, setCaption] = useState(""); // New state for the caption
  const [additionalInput, setAdditionalInput] = useState("");
  const [showThemesDropdown, setShowThemesDropdown] = useState(false);
  const [selectedTheme, setSelectedTheme] = useState(""); // New state for selected theme

  console.log("Hello from the App component!");

  //const handleTextChange = (e) => setText(e.target.value);
  //const handleImageChange = (e) => setImage(e.target.files[0]);
  const handleAdditionalInputChange = (e) => setAdditionalInput(e.target.value);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
  
    const formData = new FormData();
    formData.append("text", text); // Text prompt
    if (image) formData.append("image", image); // Uploaded image
  
    try {
      const response = await fetch("http://localhost:5000/generate", {
        method: "POST",
        body: formData,
      });
  
      const result = await response.json();
      console.log(result)
  
      if (result.success) {
        // Display the generated image
        const generatedImageBase64 = `data:image/png;base64,${result.generated_image}`;
        setOutputImage(generatedImageBase64);
        setCaption("Generated image based on your input!");
      } else {
        alert(result.message || "Failed to generate image. Please try again.");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while processing your request.");
    } finally {
      setLoading(false);
    }
  
  };

  const toggleThemesDropdown = () => {
    setShowThemesDropdown(!showThemesDropdown);
  };

  const handleThemeSelect = (theme) => {
    setSelectedTheme(theme); // Update the selected theme
    setShowThemesDropdown(false); // Close the dropdown
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <h2>Menu</h2>
        <ul>
          <li>Home</li>
          <li>Instructions</li>
          <li>FAQ</li>
          <li>Settings and Privacy</li>
        </ul>
      </div>
      <div className="content">
        <h1>Stable Diffusion Image Generator</h1>
        <form onSubmit={handleSubmit}>
        {/* Text Input */}
        <label htmlFor="text-input">Text Prompt</label>
        <input
          type="text"
          id="text-input"
          name="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          required
        />

        {/* File Input */}
        <label htmlFor="image-input">Upload Image</label>
        <input
          type="file"
          id="image-input"
          name="image"
          accept="image/*"
          onChange={(e) => setImage(e.target.files[0])}
        />

        <label htmlFor="caption-input">Input for Caption Generation:</label>
        <input
          id="caption-input"
          type="text"
          value={additionalInput}
          onChange={handleAdditionalInputChange}
          placeholder="Input for caption generation"
        />


          <button type="submit" disabled={loading}>
            {loading ? "Generating..." : "Generate Image"}
          </button>
        </form>

        {outputImage && (
          <div className="output-section">
            <h2>Generated Image:</h2>
            <img src={outputImage} alt="Generated" />
            {caption && <p className="image-caption">{caption}</p>}
          </div>
        )}

      </div>

      {/* Themes Section */}
      <div className="themes-section">
        <button onClick={toggleThemesDropdown} className="themes-button">
          Themes
        </button>
        {showThemesDropdown && (
          <ul className="themes-dropdown">
            <li onClick={() => handleThemeSelect("Semi-realistic")}>Semi-realistic</li>
            <li onClick={() => handleThemeSelect("Artistic")}>Artistic</li>
            <li onClick={() => handleThemeSelect("Atmospheric")}>Atmospheric</li>
            <li onClick={() => handleThemeSelect("Thematic")}>Thematic</li>
            <li onClick={() => handleThemeSelect("Illustrative")}>Illustrative</li>
          </ul>
        )}
        {selectedTheme && (
          <p className="selected-theme">
            Selected Theme: <strong>{selectedTheme}</strong>
          </p>
        )}
      </div>
    </div>
  );
}

export default App;

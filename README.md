# Invisible
The program uses chroma keying (similar to green screen effects) to make a specific colored cloth appear invisible by replacing it with a background image/video or other effects.
# 🧙 Ultimate Invisibility Cloak Simulation

![Demo](demo.gif)  
*Real-time invisibility effect using color masking*

A Python script that creates a Harry Potter-style "invisibility cloak" effect using computer vision. Replace any colored cloth with background images/videos or creative effects in real-time.

## ✨ Features

- **Multiple color support**: Red, green, blue, or custom HSV ranges
- **Background modes**:
  - Static image background
  - Dynamic video background
  - Blur effect
  - Neighbor fill effect
  - Classic chroma key
- **Advanced masking**:
  - Adjustable color threshold
  - Edge blending for smooth transitions
  - Mask smoothing options
- **User controls**:
  - Toggle mirror effect
  - Cycle background modes
  - Adjust settings in real-time
  - Save/load configurations
- **Debug views** for tuning effects

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/invisibility-cloak.git
cd invisibility-cloak

Install dependencies:

bash
pip install opencv-python numpy
🚀 Usage
Basic command:

bash
python Invisible.py
Common options:

bash
# With green cloak and image background
python Invisible.py --color green --image background.jpg

# With custom HSV range (Hue,Saturation,Value)
python Invisible.py --hsv-range "100-140,50-255,50-255"

# Save output to video
python Invisible.py --output result.mp4

# Load saved settings
python Invisible.py --load-config my_settings.json
⌨️ Controls while running:
Key	Function
ESC	Exit
SPACE	Cycle background modes
M	Toggle mirror effect
D	Toggle debug views
S	Save current settings
+/-	Adjust color threshold
[/]	Adjust edge blend width
⚙️ Configuration
Save your preferred settings to a JSON file:

bash
python Invisible.py --save-config my_settings.json
Example config:

json
{
    "color": "green",
    "threshold": 0.8,
    "edge_blend": 5,
    "no_mirror": false,
    "no_smoothing": false
}
📝 Customization
To modify the color detection:

Edit the get_color_range() function in Invisible.py

Add new color presets or adjust HSV ranges

Use the --hsv-range argument for precise control

🤝 Contributing
Contributions welcome! Please open an issue or PR for:

New background effects

Improved masking algorithms

Additional features

Documentation improvements

📄 License
MIT License - Free for personal and educational use

Made with magic (and OpenCV) ✨
Inspired by Harry Potter's invisibility cloak

text

This README includes:
1. Eye-catching header with emojis
2. Clear feature list
3. Installation instructions
4. Usage examples with common commands
5. Runtime controls table
6. Configuration details
7. Customization notes
8. Contribution guidelines
9. License information

You may want to:
- Add a real demo.gif showing the effect
- Include system requirements
- Add troubleshooting section if needed
- Include credits if based on other projects

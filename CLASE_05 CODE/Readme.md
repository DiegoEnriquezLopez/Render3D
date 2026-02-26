**Prerequisites:**

* Node.js installed (LTS recommended).
* VS Code installed.
* WebGPU-capable browser (Chrome/Edge latest).
* "WebGPU" VS Code extension (optional but helpful for WGSL highlighting).

---

## Part 1: Scaffolding the Project (Terminal)

1. Open **VS Code**.
2. Open the **Integrated Terminal** (`Ctrl` + `` ` ``).
3. Run:

```bash
# 1) Create project
npm create vite@latest webgpu_textured_cube -- --template vanilla-ts

# 2) Enter project folder
cd webgpu_textured_cube

# 3) Install dependencies
npm install
npm install --save-dev @webgpu/types

# 4) Start dev server
npm run dev

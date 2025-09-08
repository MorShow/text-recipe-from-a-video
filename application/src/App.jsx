import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Home from "./pages/Home";
import Recipe from "./pages/Recipe";
/* import { BrowserRouter as Router, Routes, Route } from "react-router-dom"; */

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/recipe/:recipeId" element={<Recipe />} />
      </Routes>
    </Router>
  );
}


export default App

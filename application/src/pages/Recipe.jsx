import React from "react";
/* import { useParams } from "react-router-dom"; */

const Recipe = () => {
  const { recipeId } = useParams();
  return (
    <div className="container mx-auto p-8">
      <h1 className="text-3xl font-bold mb-4">{recipeId}</h1>
      <p>This recipe page is empty for now.</p>
    </div>
  );
};

export default Recipe;
import React, { useState } from "react";
import { useForm } from "react-hook-form";
import axios from "axios";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function HealthcarePrediction() {
  const { register, handleSubmit } = useForm();
  const [result, setResult] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginId, setLoginId] = useState("");
  const [password, setPassword] = useState("");
  const [loginError, setLoginError] = useState("");

  // Handle login with static password "12345678"
  const handleLogin = (e) => {
    e.preventDefault();
    if (loginId && password === "12345678") {
      setIsLoggedIn(true);
      setLoginError("");
    } else {
      setLoginError("Invalid credentials. Please check your login ID and password.");
    }
  };

  // Handle form submission for prediction
  const onSubmit = async (data) => {
    try {
      // Ensure the patient id matches the login id
      if (data.patient_id !== loginId) {
        setLoginError("Patient ID must match the login ID.");
        return;
      }
      const response = await axios.post("http://localhost:8000/predict", data);
      setResult(response.data);
    } catch (error) {
      console.error("Prediction error", error);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold">Healthcare AI Prediction</h2>

      {!isLoggedIn ? (
        <Card>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              <Label>Login ID (Patient ID)</Label>
              <Input
                value={loginId}
                onChange={(e) => setLoginId(e.target.value)}
                required
              />
              <Label>Password</Label>
              <Input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              {loginError && <p className="text-red-600">{loginError}</p>}
              <Button type="submit" className="w-full">
                Login
              </Button>
            </form>
          </CardContent>
        </Card>
      ) : (
        <>
          <Card>
            <CardContent>
              <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
                <Label>Patient ID</Label>
                <Input {...register("patient_id")} required />

                <Label>Age</Label>
                <Input type="number" {...register("age")} required />

                <Label>Gender</Label>
                <Input {...register("gender")} required />

                <Label>Region</Label>
                <Input {...register("region")} required />

                <Label>BMI</Label>
                <Input type="number" step="0.1" {...register("bmi")} required />

                <Label>Symptoms (comma-separated)</Label>
                <Input {...register("symptoms")} required />

                <Label>Systolic BP</Label>
                <Input type="number" {...register("systolic_bp")} required />

                <Label>Diastolic BP</Label>
                <Input type="number" {...register("diastolic_bp")} required />

                <Label>Glucose</Label>
                <Input type="number" {...register("glucose")} required />

                <Label>WBC Count</Label>
                <Input type="number" step="0.1" {...register("wbc")} required />

                <Label>Outcome</Label>
                <Input {...register("outcome")} required />

                <Button type="submit" className="w-full">
                  Predict
                </Button>
              </form>
            </CardContent>
          </Card>

          {result && (
            <div className="p-4 bg-gray-100 rounded-md">
              <h3 className="text-xl font-semibold">Prediction Result</h3>
              <p>
                <strong>Patient ID:</strong> {result.patient_id}
              </p>
              <p>
                <strong>Predicted Disease:</strong> {result.predicted_disease}
              </p>
              <p>
                <strong>Next Recommendation:</strong> {result.next_recommendation}
              </p>
              <a
                href={result.neo4j_browser_url}
                className="text-blue-600 underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                View Semantic Network in Neo4j
              </a>
            </div>
          )}
        </>
      )}
    </div>
  );
}

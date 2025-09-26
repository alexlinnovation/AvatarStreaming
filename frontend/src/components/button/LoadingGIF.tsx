"use client";

export function LoadingGif({ size = 48 }: { size?: number }) {
  return (
    <img
      src="loading.gif"
      alt="Loading..."
      style={{ width: size, height: size }}
      className="inline-block"
    />
  );
}

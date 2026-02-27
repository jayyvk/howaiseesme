import { Analytics } from "@vercel/analytics/next"

export const metadata = {
  title: 'How AI Sees Me',
  description: 'Real-time CLIP vision embeddings in the browser',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body style={{ margin: 0, padding: 0 }}>{children}<Analytics/></body>
    </html>
  );
}

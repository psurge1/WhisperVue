import logo from './logo.svg';
import './App.css';
import VidStream from './templates/cam';

import { useState } from 'react';



export default function App() {
  return (
    <main>
      <h1>Whisper Vue</h1>
      <VidStream altLabel="Live Stream"/>
    </main>
  );
}



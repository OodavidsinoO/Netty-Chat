import React from "react";
import mermaid from "mermaid";

mermaid.initialize({
  startOnLoad: true,
  theme: "dark",
  suppressErrorRendering: true,
  securityLevel: "loose",
});

interface MermaidProps {
  chart: string;
}

export default class Mermaid extends React.Component<MermaidProps> {
  componentDidMount() {
    mermaid.contentLoaded();
  }
  render() {
    return <div className="mermaid">{this.props.chart}</div>;
  }
}

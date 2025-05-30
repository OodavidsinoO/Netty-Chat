"use client";
import { Footer } from "@/app/components/footer";
import { Logo } from "@/app/components/logo";
import { PresetQuery } from "@/app/components/preset-query";
import { Search } from "@/app/components/search";
import React from "react";

export default function Home() {
  return (
    <div className="absolute inset-0 min-h-[500px] flex items-center justify-center">
      <div className="relative flex flex-col gap-8 px-4 -mt-24">
        <Logo></Logo>
        <Search></Search>
        <div className="flex gap-2 flex-wrap justify-center">
          {/* Some sample questions about Computer Network */}
          <PresetQuery query="What is the difference between a hub and a switch?"></PresetQuery>
          <PresetQuery query="Explain SSL/TLS handshake process"></PresetQuery>
        </div>
        <Footer></Footer>
      </div>
    </div>
  );
}

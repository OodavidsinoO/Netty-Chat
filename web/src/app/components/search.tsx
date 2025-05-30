"use client";
import { getSearchUrl } from "@/app/utils/get-search-url";
import { ArrowRight } from "lucide-react";
import { nanoid } from "nanoid";
import { useRouter } from "next/navigation";
import React, { FC, useState } from "react";

export const Search: FC = () => {
  const [value, setValue] = useState("");
  const router = useRouter();
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (value) {
          setValue("");
          // router.push(getSearchUrl(encodeURIComponent(value), nanoid()));
          window.open(
            "/ui" + getSearchUrl(encodeURIComponent(value), nanoid()),
            "_blank",
          );
        }
      }}
    >
      <label
        className="relative bg-white flex items-center justify-center border ring-8 ring-zinc-300/20 py-2 px-2 rounded-full gap-2 focus-within:border-zinc-300"
        htmlFor="search-bar"
      >
        <input
          id="search-bar"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          autoFocus
          placeholder={
            [
              "Ask Netty AI anything ...",
              "What do you want to learn today?",
              "Search for a topic ...",
              "It's good to see you!",
              "rm -rf / --no-preserve-root... I'm joking",
            ].sort(() => Math.random() - 0.5)[0]
          }
          className="px-2 pr-6 w-full rounded-md flex-1 outline-none bg-white"
        />
        <button
          type="submit"
          className="w-auto py-1 px-2 bg-black border-black text-white fill-white active:scale-95 border overflow-hidden relative rounded-xl"
        >
          <ArrowRight size={16} />
        </button>
      </label>
    </form>
  );
};

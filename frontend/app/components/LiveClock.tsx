"use client";

import { useEffect, useState } from "react";

const FORMATTER = new Intl.DateTimeFormat("ko-KR", {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  timeZone: "Asia/Seoul",
});

export default function LiveClock() {
  const [now, setNow] = useState<Date | null>(null);

  useEffect(() => {
    const tick = () => setNow(new Date());
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  const label = now ? `${FORMATTER.format(now)} KST` : "-- : -- KST";
  return <span suppressHydrationWarning>{label}</span>;
}

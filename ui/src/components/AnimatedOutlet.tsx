import { useLocation, useOutlet } from "react-router-dom";

export function AnimatedOutlet() {
  const location = useLocation();
  const outlet = useOutlet();

  return (
    <div key={location.pathname} className="page-transition flex min-h-0 min-w-0 flex-1 flex-col">
      {outlet}
    </div>
  );
}

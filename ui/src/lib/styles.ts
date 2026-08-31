/** Shared Tailwind class strings for the React UI. */

const EASE = "duration-300 ease-[cubic-bezier(0.22,1,0.36,1)]";

export const mutedTextClass = "text-sm text-[#7c7f88]";

export const labelClass = "mb-1.5 block text-sm font-medium text-[#3b3e47]";

export const inputClass =
  "w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm text-[#12161e] shadow-sm outline-none transition focus:border-blue-800 focus:ring-2 focus:ring-blue-800/20";

export const selectClass = inputClass;

export const btnActionClass = `inline-flex items-center justify-center rounded-md border border-blue-800 bg-blue-800 px-4 py-2 text-sm font-medium text-white transition-all ${EASE} hover:border-blue-900 hover:bg-blue-900 disabled:cursor-not-allowed disabled:opacity-55`;

export const btnSecondaryClass = `inline-flex cursor-pointer items-center justify-center rounded-md border border-gray-300 bg-white/90 px-4 py-2 text-sm font-medium text-[#3b3e47] transition-all ${EASE} hover:border-gray-400 hover:bg-white disabled:cursor-not-allowed disabled:opacity-55`;

export const btnOutlineDangerClass = `inline-flex items-center justify-center rounded-md border border-red-300 bg-white text-red-700 transition-all ${EASE} hover:border-red-400 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-55`;

export const btnIconNeutralClass = `inline-flex items-center justify-center rounded-md border border-gray-300 bg-white/90 text-[#3b3e47] transition-all ${EASE} hover:border-gray-400 hover:bg-white disabled:cursor-not-allowed disabled:opacity-55`;

export const btnGroupClass =
  "mt-1.5 inline-flex overflow-hidden rounded-md border border-gray-300 bg-white shadow-sm";

export const btnGroupItemClass = `border-r border-gray-300 bg-white px-3 py-2 text-sm font-medium text-[#3b3e47] transition-all last:border-r-0 ${EASE} hover:bg-gray-50`;

export const btnGroupItemActiveClass = `border-r border-blue-800 bg-blue-800 px-3 py-2 text-sm font-medium text-white transition-all last:border-r-0 ${EASE}`;

export const alertErrorClass = "text-sm text-red-700";

export const alertWarningClass = "mt-2 text-sm text-amber-700";

export const pageActionBarClass =
  "mx-auto flex w-full max-w-3xl flex-col items-center gap-4";

export const assetItemClass =
  "rounded-md border border-gray-300 bg-white/90 p-4 shadow-sm backdrop-blur-sm";

export const sectionClass = "mx-auto w-full max-w-3xl";

export const formGridClass = "grid gap-6";

export const itemClass =
  "rounded-lg border border-gray-200 bg-white/90 p-4 shadow-sm";

export const reportAccordionClass = "report-accordion";

export const reportAccordionSummaryClass =
  "report-accordion-summary cursor-pointer list-none text-sm font-semibold text-[#12161e]";

export const progressTrackClass =
  "h-1.5 w-full overflow-hidden rounded-full bg-gray-200";

export const progressBarClass =
  "h-full rounded-full bg-blue-800 transition-[width] duration-300 ease-out";

"use client";

import Link from "next/link";


interface HeaderProps {
    onReset: () => void;
}


export function Header({ onReset }: HeaderProps) {
    const handleInputClick = () => {
        localStorage.removeItem("analysis_result");
        if (onReset) {
            onReset();
        }
    }

    return (
        <header className="archive-header">
            <div className="archive-brand">
                <span className="archive-brand-mark" />
                <div>
                    <p className="archive-brand-eyebrow">기타 이펙터</p>
                    <h1>이펙터 분석</h1>
                </div>
            </div>
            <nav className="archive-nav">
                <Link href="/input" onClick={handleInputClick}>입력</Link>
                <Link href="/overview">개요</Link>
                <Link href="/history">분석 이력</Link>
            </nav>
        </header>
    );
}
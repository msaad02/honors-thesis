const SideMenu = ({
  clearChat
}) => (
  <aside className="sidemenu">
    <div className="side-menu-button" onClick={clearChat}>
      Reset Chat
    </div>
  </aside>
);

export default SideMenu;
